import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.ingestion.document_loader import SECFilingLoader, create_sample_documents
from src.ingestion.chunker import DocumentChunker
from src.ingestion.vector_store import VectorStoreManager

def run_ingestion(data_dir=None):
    print("=" * 50)
    print("RAG COMPLIANCE — INGESTION PIPELINE")
    print("=" * 50)

    print("\n[1/4] Loading documents...")
    if data_dir:
        loader = SECFilingLoader()
        raw_docs = loader.load_from_directory(data_dir)
    else:
        print("Using built-in sample documents.")
        raw_docs = create_sample_documents()
    print(f"  Loaded {len(raw_docs)} documents")

    print("\n[2/4] Chunking...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(raw_docs)
    stats = chunker.get_chunking_stats(chunks)
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Avg chunk size: {stats['avg_chunk_size']:.0f} chars")

    print("\n[3/4] Building FAISS index (calls OpenAI API)...")
    vsm = VectorStoreManager()
    vsm.build_index(chunks)

    print("\n[4/4] Saving index...")
    vsm.save_index()

    print("\n" + "=" * 50)
    print("DONE. Now run:")
    print("  uvicorn src.api.main:app --reload --port 8000")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help="Directory with your documents")
    args = parser.parse_args()
    run_ingestion(data_dir=args.data_dir)
