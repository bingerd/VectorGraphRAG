import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from vertexai.preview.language_models import TextEmbeddingModel

from app.components import embed_text

@pytest.fixture(scope="module")
def qdrant_client():
    """Return an active Qdrant client."""
    # If `qdrant` is already initialized globally, just yield it.
    # ---------------- Local Qdrant -------------------
    QDRANT_URL = "http://localhost:6333"
    qdrant = QdrantClient(url=QDRANT_URL)
    yield qdrant

@pytest.fixture(scope="module")
def model():
    """Return an active Qdrant client."""
    # If `qdrant` is already initialized globally, just yield it.
    # ---------------- Local Qdrant -------------------
    MODEL = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

    yield MODEL

def test_qdrant_connection(qdrant_client):
    """Ensure Qdrant connection is alive."""
    health = qdrant_client.get_collections()
    assert "collections" in health.dict(), "Unable to fetch collections from Qdrant"


def test_news_collection_exists(qdrant_client):
    """Check that 'news' collection exists in Qdrant."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    assert "news" in collections, "'news' collection does not exist"


def test_collection_not_empty(qdrant_client):
    """Verify the 'news' collection contains some points."""
    info = qdrant_client.get_collection("news")
    assert info.points_count > 0, "'news' collection is empty"


def test_embedding_shape(model):
    """Ensure embedding function returns a valid vector."""
    text = "Test embedding"
    emb = embed_text(model, text)
    assert isinstance(emb, (list, tuple)), "Embedding should be list or tuple"
    assert all(isinstance(x, (float, int)) for x in emb), "Embedding should contain numbers"
    assert len(emb) > 0, "Embedding vector is empty"


def test_query_returns_results(model, qdrant_client):
    """Check that a similarity query returns at least one result."""
    query_emb = embed_text(model, "China & tariffs")
    results = qdrant_client.query_points(
        collection_name="news",
        query=query_emb,
        limit=5,
        with_payload=True
    )
    assert results is not None, "Query returned None"
    assert len(results.points) > 0, "No points returned for valid query"

    for point in results.points:
        payload = point.payload
        assert "title" in payload, "Result missing 'title'"
        assert "article_id" in payload, "Result missing 'article_id'"
        assert "chunk_id" in payload, "Result missing 'chunk_id'"
        assert isinstance(payload["title"], str), "Title should be a string"

def test_query_for_nonsense(model, qdrant_client):
    query_emb = embed_text(model, "ajskdhqwoiue")  # gibberish
    results = qdrant_client.query_points(
        collection_name="news",
        query=query_emb,
        limit=5
    )
    # Should not crash; may return fewer points
    assert results is not None

def test_similarity_scores(model, qdrant_client):
    query_emb = embed_text(model, "China & tariffs")
    results = qdrant_client.query_points(
        collection_name="news",
        query=query_emb,
        limit=3
    )
    for point in results.points:
        assert hasattr(point, "score"), "Result missing similarity score"
        assert isinstance(point.score, (float, int)), "Score is not numeric"


# # Example: Find most similar articles to a sample text
# query_text = "China & tarifs"
# query_emb = embed_text(query_text)

# results = qdrant.query_points(
#     collection_name="news",
#     query=query_emb,
#     limit=5,
#     with_payload=True
# )

# # print(results)
# print("Top 5 similar articles:")
# for res in results.points:
#     print(f"Title: {res.payload.get('title')}")
#     print(f"Article ID: {res.payload.get('article_id')}, Chunk ID: {res.payload.get('chunk_id')}")
#     print("---")
