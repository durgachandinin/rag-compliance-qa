import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from src.config import settings
from src.ingestion.vector_store import VectorStoreManager
from src.retrieval.qa_chain import ComplianceQAChain

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000,
                          description="The compliance question to answer")
    session_id: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=20)

class SourceCitation(BaseModel):
    filename: str
    category: str
    company: str = ""
    year: str = ""
    page: str = ""
    source_url: str = ""

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    source_chunks: list[str]
    processing_time_ms: float
    question: str

class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    model: str
    embedding_model: str

class AppState:
    vsm: Optional[VectorStoreManager] = None
    qa_chain: Optional[ComplianceQAChain] = None
    index_loaded: bool = False

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting RAG Compliance API...")
    app_state.vsm = VectorStoreManager()
    try:
        app_state.vsm.load_index()
        app_state.qa_chain = ComplianceQAChain(app_state.vsm)
        app_state.index_loaded = True
        print("Vector index loaded. API ready.")
    except FileNotFoundError:
        print("WARNING: No FAISS index found. Run: python scripts/run_ingestion.py")
        app_state.index_loaded = False
    yield
    print("Shutting down.")

app = FastAPI(
    title="RAG Compliance Q&A API",
    description="Q&A over SEC filings and Basel III docs using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def require_index():
    if not app_state.index_loaded or not app_state.qa_chain:
        raise HTTPException(status_code=503,
            detail="Index not loaded. Run: python scripts/run_ingestion.py")
    return app_state.qa_chain

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if app_state.index_loaded else "degraded",
        index_loaded=app_state.index_loaded,
        model=settings.OPENAI_CHAT_MODEL,
        embedding_model=settings.OPENAI_EMBEDDING_MODEL,
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, chain: ComplianceQAChain = Depends(require_index)):
    start_time = time.time()
    try:
        result = chain.ask(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    return QueryResponse(
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
        source_chunks=result["source_chunks"],
        processing_time_ms=round((time.time() - start_time) * 1000, 2),
        question=request.question,
    )

@app.post("/reset")
async def reset_conversation(chain: ComplianceQAChain = Depends(require_index)):
    chain.reset_memory()
    return {"message": "Conversation history cleared."}

@app.get("/history")
async def get_history(chain: ComplianceQAChain = Depends(require_index)):
    return {"history": chain.get_conversation_history()}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
