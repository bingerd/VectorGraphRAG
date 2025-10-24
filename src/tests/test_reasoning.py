import os
import json
import difflib
import pytest
from pydantic import BaseModel, ValidationError
from app.components import query_system

@pytest.mark.integration
def test_rag_self_eval_grounded():
    """Self-eval using live Neo4j + Qdrant, grounded in retrievable context with typed JSON responses."""

    from vertexai.generative_models import GenerativeModel
    from vertexai.preview.language_models import TextEmbeddingModel
    from qdrant_client import QdrantClient
    from neo4j import GraphDatabase
    import random
    import re

    # =====================================================
    # Typed model for JSON response from the LLM
    # =====================================================
    class GroundedQA(BaseModel):
        question: str
        expected_answer: str

    # =====================================================
    # Setup connections and models
    # =====================================================
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")

    qdrant = QdrantClient(url=QDRANT_URL)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    GENERATIVE_MODEL_NAME = os.getenv("GENERATIVE_MODEL", "gemini-2.0-flash-001")
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    llm = GenerativeModel(model_name=GENERATIVE_MODEL_NAME)

    # =====================================================
    # Select random Qdrant collection and extract context
    # =====================================================
    collections = qdrant.get_collections().collections
    if not collections:
        pytest.skip("No Qdrant collections found.")

    collection_name = random.choice(collections).name
    points, _ = qdrant.scroll(collection_name=collection_name, limit=10)
    sample_texts = [p.payload.get("content", "") for p in points if p.payload]
    context_text = "\n".join(sample_texts or "No usable context found.")

    # =====================================================
    # Generate grounded question-answer pair from LLM
    # =====================================================
    prompt = (
        "You are preparing test questions for a Retrieval-Augmented Generation (RAG) system.\n"
        "Given the following CONTEXT, generate ONE factual question that could be answered from it.\n"
        "Return valid JSON with keys 'question' and 'expected_answer'.\n"
        "Do not add explanations or markdown.\n"
        f"CONTEXT:\n{context_text}"
    )

    try:
        llm_response = llm.generate_content(
            contents=prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0,
            },
        )
        json_str = llm_response.text.strip()
        qa = GroundedQA.model_validate_json(json_str)
    except (ValidationError, json.JSONDecodeError) as e:
        # fallback: regex extract if LLM misbehaves
        print(f"[WARN] LLM JSON parse failed ({e}); applying fallback parsing.")
        text = getattr(llm_response, "text", str(llm_response))
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            pytest.skip("LLM returned invalid JSON and no recoverable structure.")
        qa = GroundedQA.model_validate_json(match.group(0))

    question = qa.question.strip()
    expected = qa.expected_answer.strip()

    if not question or not expected:
        pytest.skip(f"LLM returned incomplete data: {qa}")

    # =====================================================
    # Query RAG system
    # =====================================================
    rag_answer, _ = query_system(
        generative_model=llm,
        embedding_model=embedding_model,
        database=qdrant,
        query=question,
        driver=driver,
    )

    # =====================================================
    # Use LLM to extract the concise factual answer only
    # =====================================================

    extract_prompt = (
        "Extract ONLY the short factual answer from the following text.\n"
        "Do not include explanations, reasoning, or evidence.\n"
        "Return just the answer as plain text, nothing else.\n\n"
        f"RAG Response:\n{rag_answer}"
    )

    extract_response = llm.generate_content(
        contents=extract_prompt,
        generation_config={
            "temperature": 0,
            "max_output_tokens": 50,
        },
    )

    rag_clean = extract_response.text.strip()

    from numpy import dot
    from numpy.linalg import norm

    # === Semantic similarity (embedding-based) ===
    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    # Embed both expected and RAG answers
    expected_emb = embedding_model.get_embeddings([expected])[0].values
    rag_emb = embedding_model.get_embeddings([rag_clean])[0].values

    semantic_sim = cosine_similarity(expected_emb, rag_emb)

    print("\n===== GROUNDED SELF-EVAL =====")
    print(f"Question: {question}")
    print(f"Expected: {expected}")
    print(f"RAG Answer: {rag_answer}")
    print(f"RAG clean: {rag_clean}")
    print(f"Semantic Similarity: {semantic_sim:.2f}")

    assert semantic_sim > 0.5, f"RAG answer diverged semantically (score={semantic_sim:.2f})"

    # =====================================================
    # Evaluate similarity between expected and RAG answer
    # =====================================================
    # ratio = difflib.SequenceMatcher(None, expected.lower(), rag_answer.lower()).ratio()

    # print("\n===== GROUNDED SELF-EVAL =====")
    # print(f"Question: {question}")
    # print(f"Expected: {expected}")
    # print(f"RAG Answer: {rag_answer}")
    # print(f"Similarity: {ratio:.2f}")

    # assert ratio > 0.45, f"RAG answer diverged (similarity={ratio:.2f})"
