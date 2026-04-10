import pytest
from langchain_core.documents import Document
from src.ingestion.document_loader import SECFilingLoader, create_sample_documents
from src.ingestion.chunker import DocumentChunker
from src.retrieval.qa_chain import format_docs, extract_sources

class TestDocumentLoader:
    def test_sample_documents_returns_documents(self):
        docs = create_sample_documents()
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, Document)
            assert len(doc.page_content) > 0
            assert "document_category" in doc.metadata

    def test_infer_category_10k(self):
        loader = SECFilingLoader()
        assert loader._infer_category("goldman_sachs_10k_2023.html") == "sec_10k"

    def test_infer_category_basel(self):
        loader = SECFilingLoader()
        assert loader._infer_category("basel_iii_framework.pdf") == "basel_iii"

    def test_html_parsing_strips_tags(self):
        loader = SECFilingLoader()
        html = "<html><body><p>LCR was <b>128%</b></p><script>alert(1)</script></body></html>"
        text = loader._parse_html(html)
        assert "128%" in text
        assert "<b>" not in text
        assert "alert" not in text

class TestDocumentChunker:
    def test_chunks_are_created(self):
        docs = create_sample_documents()
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) >= len(docs)

    def test_chunks_inherit_metadata(self):
        docs = create_sample_documents()
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_documents(docs)
        for chunk in chunks:
            assert "document_category" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunking_stats(self):
        docs = create_sample_documents()
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(docs)
        stats = chunker.get_chunking_stats(chunks)
        assert stats["total_chunks"] > 0
        assert stats["avg_chunk_size"] > 0

class TestFormatDocs:
    def test_numbered_documents(self):
        docs = [Document(page_content="LCR was 128%.", metadata={"filename": "goldman.html"})]
        result = format_docs(docs)
        assert "Document 1" in result
        assert "LCR was 128%." in result

    def test_empty_returns_empty_string(self):
        assert format_docs([]) == ""

    def test_multiple_docs_separated(self):
        docs = [
            Document(page_content="A", metadata={"filename": "a.html"}),
            Document(page_content="B", metadata={"filename": "b.html"}),
        ]
        result = format_docs(docs)
        assert "Document 1" in result
        assert "Document 2" in result
        assert "---" in result

class TestExtractSources:
    def test_single_source(self):
        docs = [Document(page_content="text",
                         metadata={"filename": "goldman.html", "document_category": "sec_10k",
                                   "company": "Goldman Sachs", "year": "2023"})]
        sources = extract_sources(docs)
        assert len(sources) == 1
        assert sources[0]["filename"] == "goldman.html"

    def test_deduplicates_same_source(self):
        docs = [
            Document(page_content="chunk 1", metadata={"filename": "goldman.html", "document_category": "sec_10k"}),
            Document(page_content="chunk 2", metadata={"filename": "goldman.html", "document_category": "sec_10k"}),
        ]
        sources = extract_sources(docs)
        assert len(sources) == 1

    def test_empty_returns_empty_list(self):
        assert extract_sources([]) == []

class TestAPIModels:
    def test_query_request_validates_min_length(self):
        from src.api.main import QueryRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(question="hi")

    def test_query_request_valid(self):
        from src.api.main import QueryRequest
        req = QueryRequest(question="What is the minimum CET1 ratio under Basel III?")
        assert req.question.startswith("What")

class TestVectorStoreErrors:
    def test_search_without_index_raises(self):
        from src.ingestion.vector_store import VectorStoreManager
        vsm = VectorStoreManager.__new__(VectorStoreManager)
        vsm.vector_store = None
        with pytest.raises(ValueError, match="No index loaded"):
            vsm.similarity_search("query")

    def test_save_without_index_raises(self):
        from src.ingestion.vector_store import VectorStoreManager
        vsm = VectorStoreManager.__new__(VectorStoreManager)
        vsm.vector_store = None
        with pytest.raises(ValueError, match="No index to save"):
            vsm.save_index()

    def test_load_bad_path_raises(self):
        import os
        os.environ["OPENAI_API_KEY"] = "sk-test"
        from src.ingestion.vector_store import VectorStoreManager
        vsm = VectorStoreManager()
        with pytest.raises(FileNotFoundError):
            vsm.load_index("/nonexistent/path")

class TestEvaluator:
    def test_print_report_pass(self, capsys):
        from src.evaluation.evaluator import RAGEvaluator
        evaluator = RAGEvaluator.__new__(RAGEvaluator)
        evaluator.print_report({"answer_relevancy": 0.891, "faithfulness": 0.934,
                                 "context_recall": 0.812, "context_precision": 0.781})
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "0.891" in captured.out

    def test_print_report_needs_improvement(self, capsys):
        from src.evaluation.evaluator import RAGEvaluator
        evaluator = RAGEvaluator.__new__(RAGEvaluator)
        evaluator.print_report({"answer_relevancy": 0.70, "faithfulness": 0.934,
                                 "context_recall": 0.812, "context_precision": 0.781})
        captured = capsys.readouterr()
        assert "NEEDS IMPROVEMENT" in captured.out

    def test_eval_pairs_have_required_keys(self):
        from src.evaluation.evaluator import COMPLIANCE_EVAL_QA_PAIRS
        for pair in COMPLIANCE_EVAL_QA_PAIRS:
            assert "question" in pair
            assert "ground_truth" in pair
