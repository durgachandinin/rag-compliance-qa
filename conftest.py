import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from langchain_core.documents import Document

@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Goldman Sachs 10-K 2023. LCR was 128%. CET1 ratio was 14.5%.",
            metadata={"source": "goldman.html", "filename": "goldman.html",
                      "document_category": "sec_10k", "company": "Goldman Sachs", "year": "2023"}
        ),
        Document(
            page_content="Basel III. Minimum CET1: 4.5%. Conservation buffer: 2.5%. LCR minimum: 100%.",
            metadata={"source": "basel.pdf", "filename": "basel.pdf",
                      "document_category": "basel_iii", "issuer": "BIS", "year": "2017"}
        ),
    ]

@pytest.fixture
def chunker():
    from src.ingestion.chunker import DocumentChunker
    return DocumentChunker(chunk_size=300, chunk_overlap=50)
