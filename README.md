# LLMOps RAG integarted with VectorDB (Qdrant) & KG (Neo4j)


### Configure gcloud CLI

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and find the project ID
that we use for this training. It probably starts with `llmops-training`.
2. Run `gcloud auth login` in the terminal in your Codespace and login to your Google account.
3. Run `gcloud auth application-default login` in the terminal in your Codespace and login to your Google account.
4. Run `gcloud config set project <PROJECT_ID>` with the project ID from step 1.
5. Run `gcloud auth application-default set-quota-project <PROJECT_ID>` with the project ID from step 1.
6. Continue setting up in the [Creating personal branch](#creating-personal-branch) section below