import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from app.models.schemas import DocumentInput, DocumentChunk


class BlockType(str, Enum):
    HEADER = "header"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE = "code"
    LIST = "list"


@dataclass
class ParsedBlock:
    block_type: BlockType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    level: int = 0  # For headers


class DocumentProcessor:
    """
    Layout-Aware & Table-Preserving Document Processor.
    
    Key capabilities:
    1. Structural Block Parsing: Detects headers, markdown/pipe tables, code blocks, lists, and paragraphs.
    2. Context Breadcrumb Tracking: Tracks section hierarchies (e.g. "Item 7: MD&A > Capital Resources")
       and injects them into downstream chunks so retrieval models retain structural grounding.
    3. Header-Preserving Table Chunking: Prevents row-level fragmentation. If a table exceeds chunk size,
       splits by rows while propagating the table schema (column headers and types) to every chunk.
    4. Row-Level Semantic Linearization: Appends semantic row summaries to tabular chunks so dense
       vector embedders can match queries against tabular values without schema loss.
    5. Clean text normalization and backward-compatible recursive text chunking.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 60,
        inject_breadcrumbs: bool = True,
        linearize_tables: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.inject_breadcrumbs = inject_breadcrumbs
        self.linearize_tables = linearize_tables

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace, remove carriage returns, and clean control characters."""
        if not text:
            return ""
        text = re.sub(r'\r\n|\r', '\n', text)
        # Remove page break indicators and typical PDF footer noise
        text = re.sub(r'(?i)\n\s*---\s*page\s+\d+\s*(?:of\s+\d+)?\s*---\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def parse_blocks(self, text: str) -> List[ParsedBlock]:
        """Parse raw text into structural blocks (headers, tables, code, lists, paragraphs)."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        raw_lines = cleaned.split('\n')
        blocks: List[ParsedBlock] = []
        i = 0
        n = len(raw_lines)

        while i < n:
            line = raw_lines[i].strip()

            if not line:
                i += 1
                continue

            # 1. Fenced Code Block
            if line.startswith("```"):
                code_lines = [raw_lines[i]]
                lang = line.lstrip("`").strip()
                i += 1
                while i < n and not raw_lines[i].strip().startswith("```"):
                    code_lines.append(raw_lines[i])
                    i += 1
                if i < n:
                    code_lines.append(raw_lines[i])
                    i += 1
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.CODE,
                        content="\n".join(code_lines),
                        metadata={"language": lang or "text"}
                    )
                )
                continue

            # 2. Markdown / Pipe Table Detection
            if self._is_table_start(raw_lines, i):
                table_lines = []
                while i < n and (self._is_table_row(raw_lines[i]) or raw_lines[i].strip().startswith('|')):
                    table_lines.append(raw_lines[i].strip())
                    i += 1
                if table_lines:
                    blocks.append(
                        ParsedBlock(
                            block_type=BlockType.TABLE,
                            content="\n".join(table_lines)
                        )
                    )
                continue

            # 3. Header Detection (# Header, ## Header, or SECTION / ITEM headers)
            header_level = self._detect_header_level(line)
            if header_level > 0:
                header_title = re.sub(r'^[#\s]+', '', line).strip()
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.HEADER,
                        content=header_title,
                        level=header_level
                    )
                )
                i += 1
                continue

            # 4. List Items Detection
            if re.match(r'^(?:[-*•]|\d+[.)])\s+', line):
                list_lines = []
                while i < n and (re.match(r'^(?:[-*•]|\d+[.)])\s+', raw_lines[i].strip()) or (list_lines and raw_lines[i].startswith('   '))):
                    list_lines.append(raw_lines[i].strip())
                    i += 1
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.LIST,
                        content="\n".join(list_lines)
                    )
                )
                continue

            # 5. Regular Paragraph
            para_lines = []
            while i < n:
                curr = raw_lines[i].strip()
                if not curr:
                    break
                if curr.startswith("```") or self._is_table_start(raw_lines, i) or self._detect_header_level(curr) > 0:
                    break
                para_lines.append(curr)
                i += 1

            if para_lines:
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.PARAGRAPH,
                        content=" ".join(para_lines)
                    )
                )

        return blocks

    @staticmethod
    def _is_table_row(line: str) -> bool:
        """Check if line looks like a markdown table row with pipe delimiters."""
        s = line.strip()
        return s.startswith('|') and s.endswith('|') and len(s.split('|')) >= 3

    @classmethod
    def _is_table_start(cls, lines: List[str], index: int) -> bool:
        """Check if lines[index] begins a markdown table (header row followed by delimiter)."""
        if index + 1 >= len(lines):
            return False
        row1 = lines[index].strip()
        row2 = lines[index + 1].strip()
        if cls._is_table_row(row1):
            # Check if row2 is table delimiter (e.g., |:---|---:| or |---|---|)
            cells = [c.strip() for c in row2.split('|') if c.strip()]
            if cells and all(re.match(r'^:?-+:?$', c) for c in cells):
                return True
        return False

    @staticmethod
    def _detect_header_level(line: str) -> int:
        """Detect markdown heading level or standardized section keywords."""
        s = line.strip()
        if s.startswith('#'):
            match = re.match(r'^(#{1,6})\s+(.+)', s)
            if match:
                return len(match.group(1))
        # Detect SEC filing item headers: e.g. "ITEM 1. BUSINESS", "NOTE 8: LEASES"
        if re.match(r'^(?:ITEM|NOTE|PART|SECTION)\s+[0-9A-Z]+[.:]\s*', s, re.IGNORECASE):
            return 2
        return 0

    def _parse_table_data(self, table_text: str) -> Tuple[List[str], List[List[str]]]:
        """Extract column headers and rows from a markdown table string."""
        lines = [l.strip() for l in table_text.strip().split('\n') if l.strip()]
        if not lines:
            return [], []

        headers = [c.strip() for c in lines[0].split('|')[1:-1]]
        data_rows = []
        for line in lines[1:]:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            # Skip delimiter row
            if cells and all(re.match(r'^:?-+:?$', c) for c in cells):
                continue
            if cells:
                data_rows.append(cells)

        return headers, data_rows

    def _linearize_table_rows(self, headers: List[str], rows: List[List[str]]) -> str:
        """
        Produce a compact, natural-language representation of table rows.
        This enables dense embeddings to match contextual queries against tabular values.
        """
        narratives = []
        for row in rows:
            pairs = []
            for h, cell in zip(headers, row):
                if cell and cell != "-":
                    pairs.append(f"{h}: {cell}")
            if pairs:
                narratives.append("Row [" + "; ".join(pairs) + "]")
        return "\n".join(narratives)

    def _chunk_table(self, block: ParsedBlock, breadcrumb: str) -> List[str]:
        """
        Intelligently chunk table while preserving schema.
        If table is small, keeps whole table intact with linearization.
        If table is large, splits rows into sub-tables, repeating column headers on every chunk.
        """
        headers, rows = self._parse_table_data(block.content)
        if not headers or not rows:
            # Fallback to raw content if parsing didn't match markdown grid
            prefix = f"[Section: {breadcrumb}]\n" if breadcrumb and self.inject_breadcrumbs else ""
            return [prefix + block.content]

        header_row_str = "| " + " | ".join(headers) + " |"
        delim_row_str = "| " + " | ".join(["---"] * len(headers)) + " |"

        # Check total size
        linearized = self._linearize_table_rows(headers, rows) if self.linearize_tables else ""
        full_table_text = block.content
        if linearized:
            full_table_text += f"\n\nTabular Summary:\n{linearized}"

        prefix = f"[Section: {breadcrumb}]\n" if breadcrumb and self.inject_breadcrumbs else ""

        if len(full_table_text) + len(prefix) <= self.chunk_size or len(rows) <= 6:
            return [prefix + full_table_text]

        # Table is large: chunk rows in groups of 4-6, repeating header on every chunk
        sub_chunks = []
        batch_size = 5
        for start_idx in range(0, len(rows), batch_size):
            batch_rows = rows[start_idx : start_idx + batch_size]
            sub_table_lines = [header_row_str, delim_row_str]
            for r in batch_rows:
                # Pad row cells to match header length
                padded_r = r + [""] * (len(headers) - len(r))
                sub_table_lines.append("| " + " | ".join(padded_r[: len(headers)]) + " |")

            table_part_str = "\n".join(sub_table_lines)
            if self.linearize_tables:
                sub_lin = self._linearize_table_rows(headers, batch_rows)
                if sub_lin:
                    table_part_str += f"\n\nTabular Summary:\n{sub_lin}"

            sub_chunks.append(prefix + table_part_str)

        return sub_chunks

    def chunk_text(self, text: str) -> List[str]:
        """
        Recursive sentence-boundary text chunking for plain text (backward compatible).
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        paragraphs = cleaned.split("\n\n")
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if current_len + len(sent) + 1 > self.chunk_size and current_chunk:
                        chunk_str = " ".join(current_chunk)
                        chunks.append(chunk_str)
                        # Overlap
                        overlap_chunk = []
                        overlap_len = 0
                        for s in reversed(current_chunk):
                            if overlap_len + len(s) <= self.chunk_overlap:
                                overlap_chunk.insert(0, s)
                                overlap_len += len(s)
                            else:
                                break
                        current_chunk = overlap_chunk + [sent]
                        current_len = sum(len(s) for s in current_chunk) + len(current_chunk)
                    else:
                        current_chunk.append(sent)
                        current_len += len(sent) + 1
            else:
                if current_len + len(para) + 2 > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [para]
                    current_len = len(para)
                else:
                    current_chunk.append(para)
                    current_len += len(para) + 2

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_batch(self, documents: List[DocumentInput]) -> List[DocumentChunk]:
        """Process a list of DocumentInput instances into DocumentChunks."""
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(self.process_document(doc))
        return all_chunks

    def process_document(self, doc: DocumentInput) -> List[DocumentChunk]:
        """
        Full layout-aware ingestion: parses headers, tables, code, and paragraphs,
        maintains section hierarchy breadcrumbs, and generates indexed DocumentChunks.
        """
        doc_id = doc.id or f"doc_{uuid.uuid4().hex[:8]}"
        doc_title = doc.title or doc.metadata.get("title", f"Document {doc_id}")

        blocks = self.parse_blocks(doc.text)
        if not blocks:
            # Fallback to plain chunk_text
            plain_chunks = self.chunk_text(doc.text)
            return [
                DocumentChunk(
                    id=f"{doc_id}_chunk_{i}",
                    document_id=doc_id,
                    chunk_index=i,
                    text=c,
                    metadata={**doc.metadata, "title": doc_title, "chunk_type": "plain"}
                )
                for i, c in enumerate(plain_chunks)
            ]

        # Process blocks while maintaining section hierarchy stack
        header_stack: List[str] = [doc_title]
        raw_chunk_tuples: List[Tuple[str, str, Dict[str, Any]]] = []  # (text, chunk_type, meta)

        curr_text_buffer: List[str] = []
        curr_buffer_len = 0

        def flush_buffer():
            nonlocal curr_text_buffer, curr_buffer_len
            if curr_text_buffer:
                combined = "\n\n".join(curr_text_buffer)
                breadcrumb_str = " > ".join(header_stack)
                prefix = f"[Section: {breadcrumb_str}]\n" if self.inject_breadcrumbs else ""
                raw_chunk_tuples.append(
                    (prefix + combined, "prose", {"breadcrumb": breadcrumb_str})
                )
                curr_text_buffer = []
                curr_buffer_len = 0

        for block in blocks:
            if block.block_type == BlockType.HEADER:
                flush_buffer()
                # Adjust header stack based on level
                level = block.level
                title = block.content
                if level <= len(header_stack):
                    header_stack = header_stack[: level - 1]
                header_stack.append(title)

            elif block.block_type == BlockType.TABLE:
                flush_buffer()
                breadcrumb_str = " > ".join(header_stack)
                table_chunks = self._chunk_table(block, breadcrumb_str)
                for tc in table_chunks:
                    raw_chunk_tuples.append(
                        (tc, "table", {"breadcrumb": breadcrumb_str, "is_table": True})
                    )

            elif block.block_type == BlockType.CODE:
                flush_buffer()
                breadcrumb_str = " > ".join(header_stack)
                prefix = f"[Section: {breadcrumb_str}]\n" if self.inject_breadcrumbs else ""
                raw_chunk_tuples.append(
                    (prefix + block.content, "code", {"breadcrumb": breadcrumb_str, **block.metadata})
                )

            elif block.block_type in (BlockType.PARAGRAPH, BlockType.LIST):
                text_len = len(block.content)
                if curr_buffer_len + text_len > self.chunk_size and curr_text_buffer:
                    flush_buffer()
                curr_text_buffer.append(block.content)
                curr_buffer_len += text_len

        flush_buffer()

        # If layout parser yielded nothing (e.g. whitespace), fallback
        if not raw_chunk_tuples:
            plain_chunks = self.chunk_text(doc.text)
            raw_chunk_tuples = [(c, "prose", {}) for c in plain_chunks]

        # Construct final DocumentChunks
        doc_chunks: List[DocumentChunk] = []
        for idx, (text_content, chunk_type, extra_meta) in enumerate(raw_chunk_tuples):
            chunk_id = f"{doc_id}_chunk_{idx}"
            chunk_metadata = {
                **doc.metadata,
                "document_id": doc_id,
                "chunk_index": idx,
                "total_chunks": len(raw_chunk_tuples),
                "title": doc_title,
                "chunk_type": chunk_type,
                **extra_meta
            }
            doc_chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=doc_id,
                    chunk_index=idx,
                    text=text_content,
                    metadata=chunk_metadata
                )
            )

        return doc_chunks
