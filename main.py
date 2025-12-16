from typing import Any
import uuid
import logging

from fastapi import FastAPI
import inngest.fast_api
from dotenv import load_dotenv
from inngest import Inngest, TriggerEvent

from data_loder import load_and_chunk_pdf, embed_texts
from vectore_db import QdrantStorage
from openai import OpenAI

openai_client = OpenAI()

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

app = FastAPI()
inngest_client = Inngest(app_id="rag_app")

# --------------------------------------------------
# INGEST PDF
# --------------------------------------------------
@inngest_client.create_function(
    fn_id="rag_ingest_pdf",
    trigger=TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: Any, step: Any):

    def load_step():
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)

        chunks = load_and_chunk_pdf(pdf_path)
        return {"source_id": source_id, "chunks": chunks}

    def upsert_step(data: dict):
        chunks = data["chunks"]
        source_id = data["source_id"]

        vectors = embed_texts(chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {"text": chunks[i], "source": source_id}
            for i in range(len(chunks))
        ]

        QdrantStorage().upsert(ids, vectors, payloads)
        return {"ingested": len(chunks)}

    chunk_result = await step.run("load-and-chunk", load_step)
    return await step.run(
        "embed-and-upsert",
        lambda: upsert_step(chunk_result),
    )

# --------------------------------------------------
# QUERY PDF (RAG)
# --------------------------------------------------
@inngest_client.create_function(
    fn_id="rag_query_pdf",
    trigger=TriggerEvent(event="rag/query_pdf"),
)
async def rag_query_pdf(ctx: Any, step: Any):

    question = ctx.event.data.get("question")
    if not question:
        raise ValueError("Missing question")

    top_k = int(ctx.event.data.get("top_k", 5))

    def search_step():
        vector = embed_texts([question])[0]
        return QdrantStorage().search(vector, top_k)

    search_result = await step.run("embed-and-search", search_step)

    if not search_result["contexts"]:
        return {
            "answer": "I don’t know based on the provided context.",
            "sources": [],
            "num_contexts": 0,
        }

    context_block = "\n\n".join(search_result["contexts"])

    prompt = (
        "Answer ONLY using the context below.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )

    res = openai_client.responses.create(
        model="gpt-4o-mini",
        input=(
            "SYSTEM:\n"
            "Use ONLY the provided context. "
            "If the answer is not present, say you don't know.\n\n"
            f"USER:\n{prompt}"
        ),
        max_output_tokens=512,
    )

    answer = res.output_text.strip()

    return {
        "answer": answer,
        "sources": search_result["sources"],
        "num_contexts": len(search_result["contexts"]),
    }



# --------------------------------------------------
# FastAPI + Inngest
# --------------------------------------------------
inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf],
)
