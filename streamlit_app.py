import asyncio
import time
import os
import requests
from pathlib import Path
from contextlib import contextmanager

import streamlit as st
import inngest
from dotenv import load_dotenv

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="centered",
)

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "ingested" not in st.session_state:
    st.session_state.ingested = False
    st.session_state.source_name = None
    st.session_state.ingest_event_id = None

# --------------------------------------------------
# Spinner wrapper
# --------------------------------------------------
@contextmanager
def spinner(message: str):
    with st.spinner(message):
        yield

# --------------------------------------------------
# Async runner (fixes event loop issues)
# --------------------------------------------------
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)

# --------------------------------------------------
# Inngest client
# --------------------------------------------------
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    path = uploads_dir / file.name
    path.write_bytes(file.getbuffer())
    return path


async def send_ingest_event(pdf_path: Path) -> str:
    client = get_inngest_client()
    event_ids = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )
    return event_ids[0]


async def send_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    event_ids = await client.send(
        inngest.Event(
            name="rag/query_pdf",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return event_ids[0]


def inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_output(event_id: str, timeout_s=120, poll_interval_s=0.5) -> dict:
    start = time.time()
    last_status = None

    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}

            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Inngest run failed: {status}")

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for result (last status: {last_status})"
            )

        time.sleep(poll_interval_s)

# --------------------------------------------------
# SIDEBAR — CONTROLS
# --------------------------------------------------
with st.sidebar:
    st.header("📂 Document Control")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload once, then ask unlimited questions",
    )

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        help="Higher = more context, slower response",
    )

    if st.session_state.ingested:
        st.success("PDF ingested")
        st.caption(f"📄 {st.session_state.source_name}")

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.title("📄 RAG PDF Assistant")
st.caption("Ask questions grounded strictly in your uploaded PDF")

# --------------------------------------------------
# INGEST FLOW
# --------------------------------------------------
if uploaded_file:
    if (
        not st.session_state.ingested
        or uploaded_file.name != st.session_state.source_name
    ):
        with spinner("Uploading and ingesting PDF..."):
            saved_path = save_uploaded_pdf(uploaded_file)
            ingest_event_id = run_async(send_ingest_event(saved_path))
            time.sleep(0.3)

        st.session_state.ingested = True
        st.session_state.source_name = saved_path.name
        st.session_state.ingest_event_id = ingest_event_id

        st.success("PDF ingested successfully")
        st.caption(f"Inngest Event ID: `{ingest_event_id}`")

# --------------------------------------------------
# BLOCK QUERY UNTIL READY
# --------------------------------------------------
if not st.session_state.ingested:
    st.info("Upload a PDF from the sidebar to begin.")
    st.stop()

# --------------------------------------------------
# QUERY UI
# --------------------------------------------------
st.divider()

with st.form("query_form"):
    user_question = st.text_area(
        "Ask a question about the document",
        placeholder="e.g. Explain chapter 2 in simple terms",
        height=90,
    )
    submit = st.form_submit_button("🔍 Ask")

# --------------------------------------------------
# QUERY EXECUTION
# --------------------------------------------------
if submit and user_question.strip():
    with spinner("Searching document and generating answer..."):
        query_event_id = run_async(
            send_query_event(user_question.strip(), int(top_k))
        )
        result = wait_for_output(query_event_id)

    with st.container(border=True):
        st.subheader("🧠 Answer")
        st.write(result.get("answer", "No answer returned"))

    sources = result.get("sources", [])
    if sources:
        with st.expander("📚 Sources used"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**{i}.** {src}")

    with st.expander("⚙️ Run details"):
        st.code(
            {
                "top_k": top_k,
                "source": st.session_state.source_name,
                "query_event_id": query_event_id,
            }
        )
