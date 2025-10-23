import asyncio
from concurrent.futures import ThreadPoolExecutor

from vertexai.preview.language_models import TextEmbeddingModel
from tqdm import tqdm
import time
import uuid
import traceback

from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.asyncio import tqdm_asyncio

from pipelines.graph_ingestion import ingest_to_kg
from pipelines.vector_ingestion import ingest_to_qdrant
from pipelines.ner_inference import extract_entities_and_relations

import logging

# Create a logger for this module
logger = logging.getLogger(__name__)


# ===================================================
# --- Embedding Helpers ---
# ===================================================
def embed_text(embedding_model: TextEmbeddingModel, text: str) -> list:
    """Generate a single embedding"""
    return embedding_model.get_embeddings([text])[0].values


def embed_texts_batch(embedding_model: TextEmbeddingModel, texts: list, batch_size: int = 100) -> list:
    """Batch embeddings with retry + rate limiting"""
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            batch_embeddings = embedding_model.get_embeddings(batch)
            embeddings.extend([emb.values for emb in batch_embeddings])
        except Exception as e:
            logger.info(f"⚠️ Vertex embedding batch failed ({e}), retrying...")
            time.sleep(3)
            try:
                batch_embeddings = embedding_model.get_embeddings(batch)
                embeddings.extend([emb.values for emb in batch_embeddings])
            except Exception as e:
                logger.info(f"❌ Vertex embedding retry failed ({e})")
                embeddings.extend([None] * len(batch))
    return embeddings


# ---------------------------------------------------
# --- Async Wrappers for Blocking Functions ---
# ---------------------------------------------------
async def async_embed_texts_batch(embedding_model, chunks, batch_size):
    """Run embeddings in a background thread."""
    return await asyncio.to_thread(embed_texts_batch, embedding_model,chunks, batch_size)


async def async_extract_entities_and_relations(model, content):
    """Run LLM-based entity extraction in a background thread."""
    return await asyncio.to_thread(extract_entities_and_relations, model,content)


async def async_ingest_to_qdrant(database, article_id, title, chunks, embeddings, entities, metadata):
    return await asyncio.to_thread(
        ingest_to_qdrant, database, article_id, title, chunks, embeddings, entities, metadata
    )


async def async_ingest_to_kg(driver, article_id, title, entities, relations, metadata):
    return await asyncio.to_thread(ingest_to_kg, driver, article_id, title, entities, relations, metadata)


# ---------------------------------------------------
# --- Async Worker for One Article ---
# ---------------------------------------------------
async def process_article(model, embedding_model, database, driver, row, embed_batch_size):
    try:
        article_id = str(row.get("Unnamed: 0", uuid.uuid4()))
        title = row.get("title", "Untitled")
        content = row.get("article", "")
        metadata = {
            "author": row.get("author"),
            "date": row.get("date"),
            "section": row.get("section"),
            "publication": row.get("publication"),
        }

        # --- Split into chunks ---
        chunks = splitter.split_text(content)

        # --- Embed all chunks ---
        chunk_embeddings = await async_embed_texts_batch(embedding_model, chunks, embed_batch_size)

        # --- Extract entities & relations ---
        entities, relations = await async_extract_entities_and_relations(model, content)

        # --- Ingest into Qdrant & Neo4j (in parallel) ---
        await asyncio.gather(
            async_ingest_to_qdrant(database, article_id, title, chunks, chunk_embeddings, entities, metadata),
            async_ingest_to_kg(driver, article_id, title, entities, relations, metadata),
        )

        # logger.info(f"✅ Ingested article '{title}' ({len(chunks)} chunks)")
        return True

    except Exception as e:
        logger.info(f"⚠️ Error ingesting article '{row.get("title", "Untitled")}': {e}")
        traceback.logger.info_exc()
        return False


# ---------------------------------------------------
# --- Async Main Ingestion Function ---
# ---------------------------------------------------
async def ingest_dataframe(model, embedding_model, database, driver, df, batch_size=50, embed_batch_size=512, max_workers=10):
    """Asynchronously ingest dataframe into Qdrant + Neo4j"""
    executor = ThreadPoolExecutor(max_workers=max_workers)

    for start in tqdm(range(0, len(df), batch_size), desc="Overall Progress", unit="batch"):
        batch = df.iloc[start : start + batch_size]

        tasks = [process_article(model, embedding_model, database, driver, row, embed_batch_size) for _, row in batch.iterrows()]

        # Run articles concurrently, respecting max_workers
        semaphore = asyncio.Semaphore(max_workers)

        async def sem_task(task):
            async with semaphore:
                return await task

        results = await tqdm_asyncio.gather(
            *[sem_task(t) for t in tasks], desc=f"Batch {start // batch_size + 1}", leave=False
        )
        successes = sum(results)
        logger.info(f"✅ Completed batch {start // batch_size + 1} ({successes}/{len(batch)} succeeded)")

    executor.shutdown(wait=True)


splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
