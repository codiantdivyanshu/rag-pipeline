import fitz
import re

def text_formatter(text: str) -> str:
    try:
        return text.replace("\n", " ").strip()
    except Exception as e:
        raise ValueError(f"Text formatting failed: {str(e)}")

def extract_chunks(pdf_path):
    """
    Extract sentences chunks from PDF pages, filtering by length.
    Returns a list of dicts with page_number, sentence_chunk, chunk_token_count.
    """
    try:
        doc = fitz.open(pdf_path)
        pages_and_chunks = []
        for page_number, page in enumerate(doc):
            text = text_formatter(page.get_text())
            sentences = re.split(r'(?<=[.!?]) +', text)
            for sentence in sentences:
                clean = sentence.strip()
                if len(clean) > 30:
                    pages_and_chunks.append({
                        "page_number": page_number,
                        "sentence_chunk": clean,
                        "chunk_token_count": len(clean) / 4  # rough token count heuristic
                    })
        return pages_and_chunks
    except Exception as e:
        raise RuntimeError(f"Error extracting chunks from '{pdf_path}': {str(e)}")
