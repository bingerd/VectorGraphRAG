# Create streamlit app to query the system
import os

import streamlit as st
import vertexai
from app.components import query_system
# Local DB clients
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from vertexai.generative_models import GenerativeModel
from vertexai.preview.language_models import TextEmbeddingModel

# ===================================================
# Global models (reuse for performance)
# ===================================================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
GENERATIVE_MODEL_NAME = os.getenv("GENERATIVE_MODEL", "gemini-2.0-flash-001")
model = GenerativeModel(model_name=GENERATIVE_MODEL_NAME)


# ================== Configuration ==================

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "europe-west4")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# ---------------- Local Qdrant -------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant = QdrantClient(url=QDRANT_URL)

# Ensure collection exists
existing_collections = [c.name for c in qdrant.get_collections().collections]

# ---------------- Local Neo4j ---------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ================== Gemini Model ==================
# model = GenerativeModel(model_name="gemini-2.0-flash-001")
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

def chatbot():
    st.title("Reasoning over News Articles")
    user_question = st.text_input("Enter your question about global news:")
    if user_question:
        expansion_prompt = f"""
        Expand and rephrase the following news-related question to improve information retrieval.
        Keep it concise and factual, and return only the expanded text:

        Question: "{user_question}"
        """
        expanded_query = model.generate_content(expansion_prompt).text.strip()
        answer, full_query = query_system(
            generative_model=model,
            embedding_model=embedding_model,
            database=qdrant,
            query=expanded_query,
            driver=driver,
        )
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Full Query Details:")
        st.write(full_query)
        st.markdown("---")


# ===================================================
# Example usage
# ===================================================
# if __name__ == "__main__":
#     # Example query
#     answer, full_query = query_system(
#         "Which countries recognized Juan Guaidó as interim president in early 2019, and how did media coverage frame that recognition?"
#     )

#     print("\n--- FINAL ANSWER ---\n")
#     print(answer)
#     print("\n--- FULL PROMPT SENT TO LLM ---\n")
#     print(full_query)

#     total_duration_ms = (time.time() - START_TIME) * 1000
#     print(f"\n=== TRACE SUMMARY (Total: {round(total_duration_ms, 2)} ms) ===\n")
#     for log in TRACE_LOGS:
#         print(f"[{log['timestamp']}] {log['event']}: {log['duration_ms']} ms | {log['data']}")

#     with open("trace_log.json", "w") as f:
#         json.dump(TRACE_LOGS, f, indent=2)
