# Enterprise RAG AI Data Quality Assistant

GitHub-ready portfolio project using dummy business data, LangChain, ChromaDB, OpenAI embeddings, GPT, and Streamlit.

## What it demonstrates
- RAG (Retrieval-Augmented Generation)
- LangChain orchestration
- ChromaDB vector database
- OpenAI embeddings
- GPT response generation
- Streamlit web app
- Data quality checks on sales and inventory data

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Build vector database:

```bash
python src/build_vector_db.py
```

Run CLI:

```bash
python src/run_cli.py
```

Run web app:

```bash
streamlit run app/streamlit_app.py
```

## Interview explanation

I built a RAG assistant using LangChain, ChromaDB, OpenAI embeddings, GPT, and Streamlit. It uses dummy sales, inventory, and policy data. The system retrieves relevant business context, runs deterministic data quality checks, and uses an LLM to generate clear recommendations. This demonstrates how LLMs can be connected to enterprise data safely without fine-tuning.
