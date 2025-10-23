# ===================================================
# --- Neo4j Ingestion ---
# ===================================================
import re
from neo4j import Driver

import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

def ingest_to_kg(driver: Driver, article_id: str, title: str, entities: list,
                 relations: list, metadata: dict):
    """Insert article nodes, entities, and relations into Neo4j"""

    def _execute(tx):
        tx.run("""
            MERGE (a:Article {id: $article_id})
            SET a.title = $title,
                a.date = $date,
                a.section = $section,
                a.publication = $publication,
                a.author = $author
        """, article_id=article_id, title=title, **metadata)

        # Entities
        for ent in entities:
            tx.run("""
                MERGE (e:Entity {name: $name, type: $type})
                MERGE (a:Article {id: $article_id})
                MERGE (a)-[:MENTIONS]->(e)
            """, name=ent["name"], type=ent["type"], article_id=article_id)

        # Relations
        for rel in relations:
            rel_type = safe_rel_type(rel.get("predicate"))
            tx.run(f"""
                MERGE (s:Entity {{name: $subject}})
                MERGE (o:Entity {{name: $object}})
                MERGE (s)-[r:{rel_type}]->(o)
                SET r += $metadata
            """, subject=rel["subject"], object=rel["object"],
                   metadata=rel.get("metadata", {}))

    try:
        with driver.session() as session:
            session.execute_write(_execute)
    except Exception as e:
        logger.info(f"⚠️ Neo4j insert failed for article {article_id}: {e}")

def safe_rel_type(predicate: str) -> str:
    """Sanitize relation type for Neo4j"""
    if not predicate:
        return "RELATED_TO"
    clean = re.sub(r'[^A-Za-z0-9_]', '_', predicate.upper())
    return clean or "RELATED_TO"