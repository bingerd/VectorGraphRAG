# ===================================================
# --- Qdrant Ingestion (chunked) ---
# ===================================================
import os
import uuid
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient

import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "")

def ingest_to_qdrant(database: QdrantClient, article_id: str, title: str, chunks: list,
                     chunk_embeddings: list, entities: list, metadata: dict):
    """Insert chunked article segments into Qdrant"""
    points = []
    for i, (chunk_text, emb) in enumerate(zip(chunks, chunk_embeddings)):
        if emb is None:
            continue
        payload = {
            "article_id": article_id,
            "chunk_id": i,
            "title": title,
            "content": chunk_text,
            "entities": entities,
            **metadata
        }
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload=payload
        ))

    try:
        if points:
            database.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception as e:
        logger.info(f"⚠️ Qdrant insert failed for {title}: {e}")
