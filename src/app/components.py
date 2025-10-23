import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import spacy
from neo4j import Driver, GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi
from vertexai.generative_models import GenerativeModel
from vertexai.preview.language_models import TextEmbeddingModel

import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

# ===================================================
# Trace setup
# ===================================================
TRACE_LOGS = []
START_TIME = time.time()


def trace(event: str, start_time: float, data=None):
    """Store timing + trace entry."""
    duration_ms = (time.time() - start_time) * 1000
    TRACE_LOGS.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "duration_ms": round(duration_ms, 2),
            "data": str(data)[:400],
        }
    )


# ===================================================
# Config constants
# ===================================================
MAX_CONTEXT_CHARS = 8000
TOP_K_CHUNKS = 10  # number of chunks returned by vector search
TOP_K_ARTICLES = 10  # how many article ids to consider from KG
MAX_ENTITIES = 10
MAX_RELATIONS = 20
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")

nlp = spacy.load("en_core_web_sm")


# ===================================================
# Helper functions
# ===================================================
def embed_text(model: TextEmbeddingModel, text: str) -> list:
    t0 = time.time()
    embedding = model.get_embeddings([text])[0].values
    trace("Embedding", t0, {"text_len": len(text), "vector_dim": len(embedding)})
    return embedding


def extract_entities_and_relations(text: str):
    """Extract entities from text (used only for parsing query text here)."""
    t0 = time.time()
    doc = nlp(text)
    entities = [{"name": ent.text, "type": ent.label_} for ent in doc.ents]
    # lightweight relation extraction (not used heavily here)
    relations = []
    for sent in doc.sents:
        subj = [tok for tok in sent if tok.dep_ in ("nsubj", "nsubjpass")]
        obj = [tok for tok in sent if tok.dep_ in ("dobj", "pobj")]
        verb = [tok for tok in sent if tok.pos_ == "VERB"]
        if subj and obj and verb:
            relations.append(
                {
                    "subject": subj[0].text,
                    "predicate": verb[0].lemma_.upper().replace(" ", "_"),
                    "object": obj[0].text,
                    # "metadata": {"confidence": 0.8}
                }
            )
    trace(
        "Entity & relation extraction (query)",
        t0,
        {"entities": len(entities), "relations": len(relations)},
    )
    return entities, relations


def bm25_entity_ranking(query: str, entities: list[dict], max_entities=10):
    """Rank entities using BM25Okapi against the query."""
    if not entities:
        return []
    entity_tokens = [e["name"].lower().split() for e in entities]
    bm25 = BM25Okapi(entity_tokens)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    scored_entities = list(zip(scores, entities))
    scored_entities.sort(reverse=True, key=lambda x: x[0])
    top_entities = [e for s, e in scored_entities if s > 0][:max_entities]
    return top_entities


# ===================================================
# Formatting helpers (chunks + relations)
# ===================================================
def format_chunk_payload(payload: dict, query: str, relations: list[dict] = None) -> str:
    """
    Create a compact human-readable block for one chunk.
    payload is expected to contain: content, title, author, date, publication, entities
    relations: list of relation dicts relevant to the parent article
    """
    entities = payload.get("entities", []) or []
    if query:
        entities = bm25_entity_ranking(query, entities, MAX_ENTITIES)
    else:
        entities = entities[:MAX_ENTITIES]
    entity_names = ", ".join([e["name"] for e in entities])

    # filter relations for this article (if provided)
    relations = relations or []
    relations_filtered = [
        r
        for r in relations
        if r.get("subject") in {e["name"] for e in entities}
        or r.get("object") in {e["name"] for e in entities}
    ][:MAX_RELATIONS]
    relations_str = "; ".join(
        [f"{r['subject']} {r['predicate']} {r['object']}" for r in relations_filtered]
    )

    title = payload.get("title", "")
    date = payload.get("date", "")
    author = payload.get("author", "")
    publication = payload.get("publication", "")

    # Keep chunk content reasonably short: truncate if necessary
    content = payload.get("content", "")
    if len(content) > 2000:
        content = content[:2000] + "\n[TRUNCATED]"

    block = f"{title} {date}\n" f"{author} | {publication}\n" f"{content}\n"
    if entity_names:
        block += f"Entities: {entity_names}\n"
    if relations_str:
        block += f"Relations: {relations_str}\n"
    return block


# ===================================================
# KG helpers
# ===================================================
def kg_query_articles_by_entities(driver: GraphDatabase, entity_names: list[str]) -> list[str]:
    """
    Given entity names, return list of article_ids (distinct) that mention them.
    """
    if not entity_names:
        return []

    def _tx(tx):
        q = """
            MATCH (e:Entity)<-[:MENTIONS]-(a:Article)
            WHERE e.name IN $names
            RETURN DISTINCT a.id AS article_id
            LIMIT $limit
        """
        result = tx.run(q, names=entity_names, limit=TOP_K_ARTICLES)
        return [rec["article_id"] for rec in result]

    try:
        with driver.session() as session:
            return session.execute_read(_tx)
    except Exception as e:
        # don't fail hard; return empty
        logger.info(f"⚠️ Neo4j article query failed: {e}")
        return []


def kg_fetch_relations_for_articles(
    driver: GraphDatabase, article_ids: list[str]
) -> dict[str, list[dict]]:
    """
    Fetch relation triples from Neo4j for the provided article ids.
    Returns dict: { article_id: [ {subject, predicate, object, metadata}, ... ] }
    """
    if not article_ids:
        return {}

    def _tx(tx):
        q = """
            MATCH (a:Article)-[:MENTIONS]->(s:Entity)-[r]->(o:Entity)
            WHERE a.id IN $article_ids
            RETURN a.id AS article_id, s.name AS subject, type(r) AS predicate, o.name AS object, r AS metadata
        """
        res = tx.run(q, article_ids=article_ids)
        out = {}
        for rec in res:
            aid = rec["article_id"]
            out.setdefault(aid, []).append(
                {
                    "subject": rec["subject"],
                    "predicate": rec["predicate"],
                    "object": rec["object"],
                    "metadata": {},  # strip raw metadata if needed; keep empty or map properties
                }
            )
        return out

    try:
        with driver.session() as session:
            return session.execute_read(_tx)
    except Exception as e:
        logger.info(f"⚠️ Neo4j relations query failed: {e}")
        return {}


# ===================================================
# Context builder (chunk-aware)
# ===================================================
def fetch_and_build_context(
    driver: GraphDatabase,
    database: QdrantClient,
    combined_article_ids: list[str],
    query: str,
    post_date="2018-01-01",
) -> str:
    """
    Build context from: 1) top vector chunk hits (from query_system) and 2) extra chunks from KG article ids.
    This function expects that vector search already returned chunk-level results; but because we call vector search inside query_system,
    we re-query Qdrant here to fetch chunk points for KG articles (if necessary).
    """
    if not combined_article_ids:
        return ""

    # --- Fetch chunk points for the combined_article_ids from Qdrant ---
    # We'll retrieve up to TOP_K_CHUNKS chunks per article (but safe cap applied)
    filter_conditions = [
        FieldCondition(key="article_id", match=MatchValue(value=a_id))
        for a_id in combined_article_ids
    ]

    # scroll to gather chunk points for KG-backed articles (limit overall to avoid huge fetch)
    try:
        article_chunk_records, _ = database.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(should=filter_conditions),
            with_payload=True,
            limit=TOP_K_CHUNKS * len(combined_article_ids),
        )
    except Exception as e:
        logger.info(f"⚠️ Qdrant scroll for article chunks failed: {e}")
        article_chunk_records = []

    # Build dict of chunks by article_id
    chunks_by_article = {}
    for rec in article_chunk_records:
        try:
            aid = rec.payload.get("article_id")
            chunks_by_article.setdefault(aid, []).append(rec)
        except Exception:
            continue

    # --- Fetch relations for these articles from KG ---
    relations_by_article = kg_fetch_relations_for_articles(
        driver=driver, article_ids=combined_article_ids
    )

    # --- Assemble context parts ---
    context_parts = []
    total_len = 0

    # We iterate articles in the given order so that priority (vector hits earlier in pipeline) can be preserved by caller.
    # For each article, prefer the chunks from chunks_by_article[aid] ordered as returned by Qdrant (assumed relevance)
    for aid in combined_article_ids:
        # skip articles with date < post_date if chunk has date metadata
        article_chunks = chunks_by_article.get(aid, [])
        if not article_chunks:
            continue

        # Each chunk record is a qdrant point; payload contains chunk fields
        # sort chunks by chunk_id if present to preserve order
        try:
            article_chunks.sort(key=lambda r: r.payload.get("chunk_id", 0))
        except Exception:
            pass

        # append top chunks for that article until context limit is reached
        for rec in article_chunks:
            payload = rec.payload
            # optional date filtering
            if "date" in payload and payload["date"] < post_date:
                continue
            rels = relations_by_article.get(aid, [])
            chunk_text = format_chunk_payload(payload, query=query, relations=rels)
            if total_len + len(chunk_text) > MAX_CONTEXT_CHARS:
                # stop entirely if adding this would exceed context budget
                break
            context_parts.append(chunk_text)
            total_len += len(chunk_text)
        if total_len >= MAX_CONTEXT_CHARS:
            break

    return "\n\n".join(context_parts)


# ===================================================
# Main query pipeline (vector search over chunks + KG)
# ===================================================
def query_system(
    generative_model: GenerativeModel,
    embedding_model: TextEmbeddingModel,
    database: QdrantClient,
    query: str,
    driver: Driver,
):
    total_start = time.time()

    # --- Extract query entities ---
    t0 = time.time()
    query_entities, _ = extract_entities_and_relations(query)
    trace("Extract query entities", t0, {"entities": query_entities})

    # --- Parallel KG traversal (article ids) + embedding ---
    def kg_query():
        names = [ent["name"] for ent in query_entities]
        return kg_query_articles_by_entities(driver=driver, entity_names=names)

    t0 = time.time()
    with ThreadPoolExecutor() as executor:
        kg_future = executor.submit(kg_query)
        emb_future = executor.submit(embed_text, model=embedding_model, text=query)
        candidate_article_ids = kg_future.result()
        query_embedding = emb_future.result()
    trace("Neo4j KG traversal + Embedding", t0, {"articles_found": len(candidate_article_ids)})

    # --- Vector DB search (chunk-level) ---
    t0 = time.time()
    try:
        vector_results = database.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=TOP_K_CHUNKS,
            with_payload=True,
        )
    except Exception as e:
        logger.info(f"⚠️ Qdrant vector search failed: {e}")
        vector_results = []
    trace("Qdrant vector search (chunks)", t0, {"hits": len(vector_results)})

    # Collect article_ids from chunk hits and preserve ordering (vector hits prioritized)
    vector_article_ids_ordered = []
    vector_chunk_records = []
    for pt in vector_results:
        aid = pt.payload.get("article_id")
        if aid:
            if aid not in vector_article_ids_ordered:
                vector_article_ids_ordered.append(aid)
        vector_chunk_records.append(pt)

    # --- Combine article ids (prioritize vector hits, then KG candidates) ---
    combined_article_ids = (
        vector_article_ids_ordered
        + [cid for cid in candidate_article_ids if cid not in vector_article_ids_ordered][
            :TOP_K_ARTICLES
        ]
    )
    trace("Combine KG + vector article ids", t0, {"combined": len(combined_article_ids)})

    # --- Build context: prefer vector-returned chunks first, then KG-based chunks ---
    t0 = time.time()
    # To honor preference for vector results, we will pre-insert vector chunk records into a temporary store
    # and then fetch additional chunks for KG-only articles inside fetch_and_build_context.
    # We'll create a small temporary context using the vector_chunk_records first.
    context_parts = []
    total_len = 0

    # Fetch relations for combined articles (will be used for both vector chunks & KG chunks)
    relations_by_article = kg_fetch_relations_for_articles(
        driver=driver, article_ids=combined_article_ids
    )

    # Add vector chunk records to context first (they are most relevant)
    for rec in vector_chunk_records:
        aid = rec.payload.get("article_id")
        # optional date filter
        # if "date" in rec.payload and rec.payload["date"] < "2018-01-01":
        #     continue

        chunk_block = format_chunk_payload(
            rec.payload, query=query, relations=relations_by_article.get(aid, [])
        )
        if total_len + len(chunk_block) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(chunk_block)
        total_len += len(chunk_block)

    # If we still have room, fetch more chunks for KG-derived articles not already covered by vector hits
    remaining_article_ids = [
        aid
        for aid in combined_article_ids
        if aid not in {r.payload.get("article_id") for r in vector_chunk_records}
    ]
    if remaining_article_ids and total_len < MAX_CONTEXT_CHARS:
        # fetch additional chunks via fetch_and_build_context but only for remaining_article_ids
        extra_context = fetch_and_build_context(
            driver=driver,
            database=database,
            combined_article_ids=remaining_article_ids,
            query=query,
        )
        if extra_context:
            # ensure we don't exceed MAX_CONTEXT_CHARS
            if total_len + len(extra_context) > MAX_CONTEXT_CHARS:
                extra_context = extra_context[: (MAX_CONTEXT_CHARS - total_len)] + "\n[TRUNCATED]"
            context_parts.append(extra_context)
            total_len += len(extra_context)

    context_text = "\n\n".join(context_parts)
    trace("Build context text", t0, {"context_len": len(context_text)})

    # --- Generate final response ---
    t0 = time.time()
    # Structured prompt: separate "Relevant snippets" and "Entity/relations summary"
    entity_summary_lines = []
    for aid in combined_article_ids:
        rels = relations_by_article.get(aid, [])
        if not rels:
            continue
        # compact summary per article (max few relations)
        rels_short = rels[:5]
        s = f"Article {aid}: " + "; ".join(
            [f"{r['subject']} -[{r['predicate']}]-> {r['object']}" for r in rels_short]
        )
        entity_summary_lines.append(s)
    entity_summary = "\n".join(entity_summary_lines)

    full_query = (
        """
    You are a retrieval-augmented assistant with two data sources:
    1. Knowledge graph facts (entity relationships)
    2. Document snippets (retrieved by semantic similarity)

    Use only this information to answer the query.
    If the answer isn’t explicitly supported, say:
    "I don’t know based on the provided information."

    User query:
    {query}

    Knowledge graph facts (triplet form):
    {entity_summary}

    Document snippets (ranked by relevance):
    {context_text}

    Respond concisely in this format:
    Answer: <factual answer>
    Evidence:
    - <doc title / author / publication> \n
    - <relevant KG relation> \n
    """
    ).format(query=query, entity_summary=entity_summary, context_text=context_text)

    try:
        response = generative_model.generate_content(full_query)
        llm_text = response.text
    except Exception as e:
        logger.info(f"⚠️ LLM generation failed: {e}")
        llm_text = "Error generating response."

    trace("Gemini LLM generation", t0, {"response_len": len(llm_text)})

    trace("Total query runtime", total_start)
    return llm_text, full_query
