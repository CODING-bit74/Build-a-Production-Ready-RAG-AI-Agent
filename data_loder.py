from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from pathlib import Path
from typing import List

load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

def load_and_chunk_pdf(path: str) -> List[str]:
    pdf_path = Path(path).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    docs = PDFReader().load_data(file=pdf_path)

    texts = [d.text.strip() for d in docs if d.text and d.text.strip()]
    if not texts:
        raise ValueError("No extractable text found in PDF")

    chunks: List[str] = []
    for text in texts:
        chunks.extend(splitter.split_text(text))

    if not chunks:
        raise ValueError("Chunking produced no text")

    return chunks

def embed_texts(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    if not texts:
        raise ValueError("embed_texts received empty input")

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])

    if len(all_embeddings) != len(texts):
        raise RuntimeError("Embedding count mismatch")

    if len(all_embeddings[0]) != EMBED_DIM:
        raise RuntimeError("Embedding dimension mismatch")

    return all_embeddings
