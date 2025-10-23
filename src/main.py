import asyncio

import kagglehub
import nest_asyncio
import pandas as pd
from pathlib import Path
# Create streamlit app to query the system
import os

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview.language_models import TextEmbeddingModel
# Local DB clients
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# ================== Configuration ==================

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "europe-west4")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# ---------------- Local Qdrant -------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant = QdrantClient(url=QDRANT_URL)

# ---------------- Local Neo4j ---------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

from pipelines.document_ingestion import ingest_dataframe

# Download latest version
PATH = kagglehub.dataset_download("davidmckinley/all-the-news-dataset")


# ---------------------------------------------------
# --- Entry Point ---
# ---------------------------------------------------
# Determine if script is being run directly
# CLI: <name> pipeline run 
# Starts a ingestion pipeline for news articles
# CLI: <name> app run
# ARGS: batch-size, embed-batch-size, max-workers, sample-size

# Starts the Streamlit app for querying
# Using Click's command line interface
import click  # type: ignore

@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "--run_date", 
    required=True,
    help="Date for which to run the pipeline in YYYY-MM-DD format.")
@click.option("--sample_size", type=int, default=100)
@click.option("--batch_size", type=int, default=50)
@click.option("--embed_batch_size", type=int, default=512)
@click.option("--max_workers", type=int, default=10)
def pipeline(run_date, sample_size, batch_size, embed_batch_size, max_workers):
    """Starts a ingestion pipeline for news articles."""
    model = GenerativeModel(model_name="gemini-2.0-flash-001")
    embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    # df = pd.read_csv(PATH + "/all-the-news-2-1.csv", skiprows=1000000, nrows=1500000)

    base_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    project_root = base_dir
    data_path = project_root / "data" / run_date 

    csv_files = sorted(data_path.glob("news_articles_*.csv"))[0]
    df = pd.read_csv(csv_files)
    nest_asyncio.apply()  # allows nested asyncio loops in notebooks

    asyncio.run(ingest_dataframe(model, embedding_model, qdrant, driver, df, batch_size, embed_batch_size, max_workers))

@cli.command()
def app():
    """Starts the Streamlit app for querying."""
    from app.app import chatbot

    chatbot()

if __name__ == "__main__":
    cli()
