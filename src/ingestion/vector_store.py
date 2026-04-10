from typing import List, Optional
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import settings

class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.vector_store: Optional[FAISS] = None

    def build_index(self, chunks: List[Document]) -> FAISS:
        print(f"Building FAISS index for {len(chunks)} chunks...")
        self.vector_store = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        print(f"Index built. {len(chunks)} vectors stored.")
        return self.vector_store

    def save_index(self, path: str = None) -> None:
        if not self.vector_store:
            raise ValueError("No index to save. Call build_index() first.")
        save_path = path or settings.FAISS_INDEX_PATH
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(save_path)
        print(f"Index saved to {save_path}")

    def load_index(self, path: str = None) -> FAISS:
        load_path = path or settings.FAISS_INDEX_PATH
        if not Path(load_path).exists():
            raise FileNotFoundError(f"No FAISS index at {load_path}. Run ingestion first.")
        self.vector_store = FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"Index loaded from {load_path}")
        return self.vector_store

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        if not self.vector_store:
            raise ValueError("No index loaded. Call build_index() or load_index() first.")
        return self.vector_store.similarity_search(query, k=k or settings.TOP_K_RESULTS)

    def similarity_search_with_scores(self, query: str, k: int = None) -> List[tuple]:
        if not self.vector_store:
            raise ValueError("No index loaded.")
        return self.vector_store.similarity_search_with_score(query, k=k or settings.TOP_K_RESULTS)

    def get_retriever(self, k: int = None):
        if not self.vector_store:
            raise ValueError("No index loaded.")
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k or settings.TOP_K_RESULTS},
        )
