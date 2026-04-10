from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from src.config import settings
from src.ingestion.vector_store import VectorStoreManager

SYSTEM_PROMPT = (
    "You are a regulatory compliance expert specializing in SEC filings "
    "and Basel III banking regulations. Answer questions accurately using "
    "ONLY the provided document excerpts below.\n\n"
    "RULES:\n"
    "1. Base your answer exclusively on the provided context. Do not use prior knowledge.\n"
    "2. If the context is insufficient, say: 'The provided documents do not contain "
    "enough information to answer this question.'\n"
    "3. Always cite your sources by referencing the document name and relevant details.\n"
    "4. Be precise with numerical data — ratios, percentages, and capital amounts.\n"
    "5. If multiple documents are relevant, synthesize them into a coherent answer.\n\n"
    "RETRIEVED CONTEXT:\n{context}"
)

COMPLIANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return ""
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source_label = meta.get("filename", meta.get("source", "Unknown"))
        parts = [f"Document {i}: {source_label}"]
        if meta.get("company"):
            parts.append(f"Company: {meta['company']}")
        if meta.get("year"):
            parts.append(f"Year: {meta['year']}")
        if meta.get("document_category"):
            parts.append(f"Type: {meta['document_category']}")
        if meta.get("page"):
            parts.append(f"Page: {meta['page']}")
        header = " | ".join(parts)
        formatted.append(f"[{header}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def extract_sources(docs: List[Document]) -> List[dict]:
    seen = set()
    sources = []
    for doc in docs:
        meta = doc.metadata
        key = meta.get("filename", meta.get("source", "unknown"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": key,
                "category": meta.get("document_category", "unknown"),
                "company": meta.get("company", ""),
                "year": meta.get("year", ""),
                "page": str(meta.get("page", "")),
                "source_url": meta.get("source_url", ""),
            })
    return sources


class ComplianceQAChain:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vsm = vector_store_manager
        self.chat_history: List[BaseMessage] = []
        self.llm = ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.retriever = self.vsm.get_retriever(k=settings.TOP_K_RESULTS)
        self.output_parser = StrOutputParser()
        self.chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(
                    lambda x: format_docs(self.retriever.invoke(x["question"]))
                )
            )
            | COMPLIANCE_PROMPT
            | self.llm
            | self.output_parser
        )

    def ask(self, question: str) -> dict:
        retrieved_docs = self.retriever.invoke(question)
        sources = extract_sources(retrieved_docs)
        source_chunks = [doc.page_content[:500] for doc in retrieved_docs]
        answer = self.chain.invoke({
            "question": question,
            "chat_history": self._get_windowed_history(),
        })
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        return {"answer": answer, "sources": sources, "source_chunks": source_chunks}

    def _get_windowed_history(self) -> List[BaseMessage]:
        return self.chat_history[-(settings.MEMORY_WINDOW * 2):]

    def reset_memory(self) -> None:
        self.chat_history = []

    def get_conversation_history(self) -> List[dict]:
        return [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in self.chat_history
        ]
