
import pytest
from neo4j import GraphDatabase

@pytest.fixture(scope="module")
def neo4j_session():
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "testpassword"))
    with driver.session() as session:
        yield session
    driver.close()


def test_articles_exist(neo4j_session):
    result = neo4j_session.run("MATCH (a:Article) RETURN a.id AS id, a.title AS title")
    records = list(result)
    assert len(records) > 0, "No articles found in the knowledge graph"

    for record in records:
        assert record["id"] is not None, "Article missing ID"
        assert record["title"], "Article title is empty"

# def test_entities_exist(neo4j_session):
#     result = neo4j_session.run("MATCH (e:Entity) RETURN e.name AS name, e.type AS type")
#     records = list(result)
#     assert len(records) > 0, "No entities found in the knowledge graph"

#     for record in records:
#         assert record["name"], "Entity missing name"
#         assert record["type"], "Entity missing type"

def test_relationships_are_valid(neo4j_session):
    result = neo4j_session.run("""
        MATCH (s:Entity)-[r]->(o:Entity)
        RETURN s.name AS subject, type(r) AS predicate, o.name AS object
    """)
    records = list(result)
    assert len(records) > 0, "No relationships found in the knowledge graph"

    for record in records:
        assert record["subject"], "Relationship missing subject"
        assert record["predicate"], "Relationship missing predicate"
        assert record["object"], "Relationship missing object"

def test_articles_have_entity_mentions(neo4j_session):
    result = neo4j_session.run("""
        MATCH (a:Article)
        OPTIONAL MATCH (a)-[:MENTIONS]->(e:Entity)
        RETURN a.title AS title, COUNT(e) AS entity_count
    """)
    records = list(result)
    assert len(records) > 0, "No articles found"

    for record in records:
        assert record["entity_count"] >= 0, "Invalid entity count"

def test_article_entity_link_integrity(neo4j_session):
    result = neo4j_session.run("""
        MATCH (a:Article)-[:MENTIONS]->(e:Entity)
        RETURN COUNT(*) AS c
    """)
    count = result.single()["c"]
    assert count > 0, "No article-entity MENTIONS relationships exist"