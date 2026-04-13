# RAG Pipeline for Regulatory Compliance Q&A

Q&A over SEC filings and Basel III documents using Retrieval-Augmented Generation.
89% answer relevance evaluated via RAGAS.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env        # add your OPENAI_API_KEY
python scripts/run_ingestion.py
uvicorn src.api.main:app --reload --port 8000
# open http://localhost:8000/docs
```

## Stack
- LangChain LCEL + FAISS + OpenAI embeddings
- FastAPI with conversation memory and source citation
- RAGAS evaluation framework

## Data

Real regulatory documents ingested:
- Basel III framework documents (BIS/BCBS PDFs)
- Place PDFs in `data/raw/` and run `python scripts/run_ingestion.py` to rebuild the index
