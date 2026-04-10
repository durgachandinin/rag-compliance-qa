from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

class DocumentChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        for doc_idx, doc in enumerate(documents):
            chunks = self.splitter.split_documents([doc])
            for chunk_idx, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": chunk_idx,
                    "total_chunks_in_doc": len(chunks),
                    "parent_doc_index": doc_idx,
                    "chunk_char_length": len(chunk.page_content),
                })
                all_chunks.append(chunk)
        return all_chunks

    def get_chunking_stats(self, chunks: List[Document]) -> dict:
        lengths = [len(c.page_content) for c in chunks]
        categories = {}
        for chunk in chunks:
            cat = chunk.metadata.get("document_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(lengths) / len(lengths) if lengths else 0,
            "min_chunk_size": min(lengths) if lengths else 0,
            "max_chunk_size": max(lengths) if lengths else 0,
            "chunks_by_category": categories,
        }
