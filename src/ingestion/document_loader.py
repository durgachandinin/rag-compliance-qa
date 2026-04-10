import os
import requests
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_core.documents import Document
from tqdm import tqdm

class SECFilingLoader:
    def load_from_directory(self, directory: str) -> List[Document]:
        docs = []
        path = Path(directory)
        files = list(path.glob("**/*"))
        supported = [f for f in files if f.suffix.lower() in {".pdf", ".html", ".htm", ".txt"}]
        print(f"Found {len(supported)} documents to load...")
        for file_path in tqdm(supported, desc="Loading documents"):
            try:
                if file_path.suffix.lower() == ".pdf":
                    docs.extend(self._load_pdf(file_path))
                elif file_path.suffix.lower() in {".html", ".htm"}:
                    docs.extend(self._load_html(file_path))
                elif file_path.suffix.lower() == ".txt":
                    docs.extend(self._load_txt(file_path))
            except Exception as e:
                print(f"  Warning: Could not load {file_path.name}: {e}")
        print(f"Loaded {len(docs)} document pages total.")
        return docs

    def _load_pdf(self, file_path: Path) -> List[Document]:
        reader = PdfReader(str(file_path))
        docs = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": str(file_path), "filename": file_path.name,
                               "page": page_num + 1, "source_type": "pdf",
                               "document_category": self._infer_category(file_path.name)}
                ))
        return docs

    def _load_html(self, file_path: Path) -> List[Document]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        text = self._parse_html(html)
        return [Document(page_content=text,
                         metadata={"source": str(file_path), "filename": file_path.name,
                                   "source_type": "html",
                                   "document_category": self._infer_category(file_path.name)})]

    def _load_txt(self, file_path: Path) -> List[Document]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [Document(page_content=text,
                         metadata={"source": str(file_path), "filename": file_path.name,
                                   "source_type": "txt",
                                   "document_category": self._infer_category(file_path.name)})]

    def _parse_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _infer_category(self, filename: str) -> str:
        name_lower = filename.lower()
        if any(x in name_lower for x in ["10-k", "10k", "annual"]):
            return "sec_10k"
        elif any(x in name_lower for x in ["10-q", "10q", "quarterly"]):
            return "sec_10q"
        elif any(x in name_lower for x in ["8-k", "8k"]):
            return "sec_8k"
        elif any(x in name_lower for x in ["basel", "bis", "bcbs"]):
            return "basel_iii"
        else:
            return "regulatory_other"


def create_sample_documents() -> List[Document]:
    return [
        Document(
            page_content="""GOLDMAN SACHS GROUP INC — ANNUAL REPORT 2023 (10-K)
RISK FACTORS — LIQUIDITY RISK
We maintain liquidity to meet our funding needs under a range of stressed market conditions.
Our Global Core Liquid Assets (GCLA) averaged $433 billion during 2023, compared to $418 billion in 2022.
Under Basel III requirements, we are required to maintain a Liquidity Coverage Ratio (LCR) of at least 100%.
Our LCR was approximately 128% as of December 2023.
CAPITAL ADEQUACY
Our Common Equity Tier 1 (CET1) capital ratio under the Standardized approach was 14.5%
as of December 31, 2023, compared to the regulatory minimum of 4.5% plus our stress capital
buffer of 6.4%, for a total requirement of 10.9%.""",
            metadata={"source": "data/raw/goldman_sachs_10k_2023.html", "filename": "goldman_sachs_10k_2023.html",
                      "document_category": "sec_10k", "company": "Goldman Sachs", "year": "2023", "source_type": "html"}
        ),
        Document(
            page_content="""BASEL III: A GLOBAL REGULATORY FRAMEWORK FOR MORE RESILIENT BANKS
Bank for International Settlements — Basel Committee on Banking Supervision
MINIMUM CAPITAL REQUIREMENTS
Banks must maintain the following minimum capital ratios at all times:
4.5% Common Equity Tier 1 (CET1) to risk-weighted assets
6.0% Tier 1 capital to risk-weighted assets
8.0% Total capital to risk-weighted assets
CAPITAL CONSERVATION BUFFER
Banks must also maintain a capital conservation buffer of 2.5%, comprised of CET1 capital,
above the regulatory minimum capital requirements.
LIQUIDITY STANDARDS
The Liquidity Coverage Ratio (LCR) requires banks to hold sufficient High Quality Liquid
Assets (HQLA) to survive a significant stress scenario lasting 30 calendar days.
The Net Stable Funding Ratio (NSFR) promotes resilience over a one-year time horizon.""",
            metadata={"source": "data/raw/basel_iii_framework.pdf", "filename": "basel_iii_framework.pdf",
                      "document_category": "basel_iii", "issuer": "BIS/BCBS", "year": "2017", "source_type": "pdf"}
        ),
        Document(
            page_content="""JPMORGAN CHASE & CO — FORM 10-Q Q3 2023
CAPITAL MANAGEMENT
CET1 capital ratio: 14.3% (September 30, 2023)
Tier 1 capital ratio: 15.9%
Total capital ratio: 17.5%
STRESS CAPITAL BUFFER
The Federal Reserve's stress capital buffer requirement for JPMorgan Chase is 2.5%.
As a Global Systemically Important Bank (G-SIB), we are also subject to a G-SIB surcharge of 3.5%.
Our effective CET1 minimum requirement is therefore 10.5%.""",
            metadata={"source": "data/raw/jpmorgan_10q_q3_2023.html", "filename": "jpmorgan_10q_q3_2023.html",
                      "document_category": "sec_10q", "company": "JPMorgan Chase", "year": "2023",
                      "quarter": "Q3", "source_type": "html"}
        ),
    ]
