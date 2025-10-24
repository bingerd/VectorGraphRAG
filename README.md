# LLMOps RAG integarted with VectorDB (Qdrant) & KG (Neo4j)

This project is based on the `All the News` dataset on Kaggle[url/https://www.kaggle.com/datasets/davidmckinley/all-the-news-dataset].
The articles can be dated from early 2016 until late 2019. 



### Configure gcloud CLI

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and find the project ID
that we use for this training. It probably starts with `llmops-training`.
2. Run `gcloud auth login` in the terminal in your Codespace and login to your Google account.
3. Run `gcloud auth application-default login` in the terminal in your Codespace and login to your Google account.
4. Run `gcloud config set project <PROJECT_ID>` with the project ID from step 1.
5. Run `gcloud auth application-default set-quota-project <PROJECT_ID>` with the project ID from step 1.
6. Continue setting up in the [Creating personal branch](#creating-personal-branch) section below

### Spinning up Qdrant & Neo4j databases

1. Run `docker-compose up -d` in terminal
2. Check Qdrant with this URL: `http://localhost:6333/dashboard#/welcome`
3. Check Neo4j with this URL: `https://browser.neo4j.io/`


### Ingesting documents and NER

The raw data is too high to fit into memory (~8.8GB unpacked CSV). Therefore, the data must be chunked using `notebooks/partition_data.ipynb`. This notebook will partition the articles by date.

Next, we can ingest a single day of data by calling `python -m src.main pipeline --run-date="2016-01-01"` (YYYY-MM-DD).

### Demo

Open the streamlit dashboard using `streamlit run src.main.py app` and ask your news-related question!
(Make sure docker deamon is running and Qdrant/Neo4j are up!)

### Running tests

`pytest -s` 