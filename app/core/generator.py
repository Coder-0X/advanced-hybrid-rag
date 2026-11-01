import os
import re
from typing import List, Tuple, Optional
from app.models.schemas import RetrievedChunk

class GroundedRAGGenerator:
    """
    Synthesizes answers strictly grounded in retrieved context passages,
    reducing hallucination with citation tagging and attribution verification.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    def generate(self, query: str, context_chunks: List[RetrievedChunk]) -> Tuple[str, float]:
        """
        Generate answer based on context_chunks.
        Returns: (answer_text, grounding_score)
        """
        if not context_chunks:
            return "No relevant context found to answer the query.", 0.0

        # Build numbered context passages with citations
        context_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            title = chunk.metadata.get("title", f"Document {chunk.document_id}")
            context_blocks.append(f"[{idx}] (Source: {title}):\n{chunk.text.strip()}")

        context_str = "\n\n".join(context_blocks)

        # 1. Check Gemini Provider
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                prompt = (
                    "You are a strict, faithful question-answering assistant.\n"
                    "Answer the user query ONLY using the provided context passages below.\n"
                    "Cite sources using [1], [2], etc.\n"
                    "Do NOT extrapolate or assume information not present in the passages.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Query: {query}\n\nAnswer:"
                )
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                answer = response.text.strip()
                grounding_score = self.verify_grounding(answer, [c.text for c in context_chunks])
                return answer, grounding_score
            except Exception:
                pass

        # 2. Check OpenAI Provider
        if os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                client = openai.OpenAI()
                prompt = (
                    "You are a strict, faithful question-answering assistant.\n"
                    "Answer the user query ONLY using the provided context passages below.\n"
                    "Cite sources using [1], [2], etc.\n"
                    "Do NOT extrapolate or assume information not present in the passages.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Query: {query}\n\nAnswer:"
                )
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                answer = response.choices[0].message.content.strip()
                grounding_score = self.verify_grounding(answer, [c.text for c in context_chunks])
                return answer, grounding_score
            except Exception:
                pass

        # Standalone High-Precision Extractive-Synthesis Generator (zero-dependency, deterministic)
        answer = self._synthesize_grounded_answer(query, context_chunks)
        grounding_score = self.verify_grounding(answer, [c.text for c in context_chunks])
        return answer, grounding_score

    def _synthesize_grounded_answer(self, query: str, context_chunks: List[RetrievedChunk]) -> str:
        """
        Deterministic, faithful synthesis from top retrieved chunks.
        Extracts the most salient sentences answering the query and formats with citations.
        """
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        selected_statements = []
        for idx, chunk in enumerate(context_chunks, start=1):
            sentences = re.split(r'(?<=[.!?])\s+', chunk.text)
            best_sentence = None
            best_score = -1

            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 15:
                    continue
                s_words = set(re.findall(r'\b\w{3,}\b', s_clean.lower()))
                overlap = len(query_words.intersection(s_words))
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = s_clean

            if best_sentence and best_score > 0:
                selected_statements.append((best_score, f"{best_sentence} [{idx}]"))

        # Sort by relevance to query
        selected_statements.sort(key=lambda x: x[0], reverse=True)
        
        if not selected_statements:
            # Fallback to top chunk snippet
            first_chunk = context_chunks[0]
            first_sentence = first_chunk.text.split(". ")[0] + "."
            return f"{first_sentence} [1]"

        # Deduplicate and synthesize top 2-3 sentences
        seen = set()
        final_lines = []
        for _, stmt in selected_statements[:3]:
            simplified = re.sub(r'\[\d+\]', '', stmt).strip().lower()
            if simplified not in seen:
                seen.add(simplified)
                final_lines.append(stmt)

        return " ".join(final_lines)

    @staticmethod
    def verify_grounding(answer: str, context_texts: List[str]) -> float:
        """
        Measures the fraction of key terms / n-grams in the answer that are attested in the context.
        Formula: Attested_Content_Words / Total_Content_Words in Answer
        """
        if not answer or not context_texts:
            return 0.0

        all_context = " ".join(context_texts).lower()
        # Extract content words (length >= 4) from the answer
        answer_words = re.findall(r'\b[a-zA-Z]{4,}\b', answer.lower())
        if not answer_words:
            return 1.0

        attested_count = sum(1 for word in answer_words if word in all_context)
        return round(attested_count / len(answer_words), 4)
