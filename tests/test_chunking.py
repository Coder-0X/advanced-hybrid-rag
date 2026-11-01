import pytest
from app.core.document_processor import DocumentProcessor
from app.models.schemas import DocumentInput


def test_chunk_text_small():
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    text = "Short text under the threshold."
    chunks = processor.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_multi_sentence():
    processor = DocumentProcessor(chunk_size=80, chunk_overlap=15)
    text = (
        "Sentence one is simple. Sentence two adds more context. "
        "Sentence three continues the explanation. Sentence four concludes it."
    )
    chunks = processor.chunk_text(text)
    assert len(chunks) > 1
    # Check all sentences are retained
    full_reconstructed = " ".join(chunks)
    assert "Sentence one" in full_reconstructed
    assert "Sentence four" in full_reconstructed


def test_process_document_metadata():
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    doc = DocumentInput(
        id="test_doc_01",
        title="Test Title",
        text="First sentence here. Second sentence follows. Third sentence wraps up.",
        metadata={"author": "Alice"}
    )
    chunks = processor.process_document(doc)
    assert len(chunks) >= 1
    assert chunks[0].document_id == "test_doc_01"
    assert chunks[0].metadata["author"] == "Alice"
    assert chunks[0].metadata["title"] == "Test Title"


def test_table_parsing_and_schema_preservation():
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    doc_text = """
# Financial Position
Below is the balance sheet summary for the fiscal year:

| Metric | 2024 (USD M) | 2025 (USD M) |
|---|---|---|
| Cash and Cash Equivalents | 12450 | 14800 |
| Total Current Assets | 34500 | 41200 |
| Total Current Liabilities | 15200 | 17900 |

Additional notes on accounting principles follow.
"""
    doc = DocumentInput(id="doc_fin_01", title="Annual Report", text=doc_text)
    chunks = processor.process_document(doc)

    # Find the table chunk
    table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
    assert len(table_chunks) >= 1
    table_chunk = table_chunks[0]

    # Check table structure is intact
    assert "| Metric |" in table_chunk.text
    assert "Cash and Cash Equivalents" in table_chunk.text
    assert "14800" in table_chunk.text
    # Check linearization summary is present
    assert "Tabular Summary:" in table_chunk.text
    assert "Metric: Cash and Cash Equivalents" in table_chunk.text
    # Check breadcrumb includes section header
    assert "Financial Position" in table_chunk.metadata.get("breadcrumb", "")


def test_large_table_header_propagation():
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=30)
    # Generate a table with 12 rows that will trigger multi-chunking
    rows = "\n".join([f"| Metric_{i} | Value_{i} | Status_{i} |" for i in range(12)])
    doc_text = f"""
# Enterprise Inventory
| Component | Metric | Status |
|---|---|---|
{rows}
"""
    doc = DocumentInput(id="doc_large_table", title="Inventory Spec", text=doc_text)
    chunks = processor.process_document(doc)
    table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]

    assert len(table_chunks) >= 2
    for tc in table_chunks:
        # Every chunk must retain the column header row
        assert "| Component | Metric | Status |" in tc.text
        assert "| --- | --- | --- |" in tc.text
        assert tc.metadata.get("is_table") is True
