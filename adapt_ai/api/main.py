"""FastAPI application — thin HTTP wrapper over the LangGraph pipeline."""
from __future__ import annotations
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from adapt_ai.orchestrator.client import build_mcp_client, MCPClient
from adapt_ai.orchestrator.session import SessionManager
from adapt_ai.agents.graph import build_graph
from adapt_ai.llmops.tracing import setup_tracing
from adapt_ai.domain.patient_handler import PatientHandler

_UI_DIR = Path(__file__).parent.parent.parent / "ui"

logger = logging.getLogger(__name__)

# ── Shared state (initialised at startup) ─────────────────────────────────────
_mcp_client: MCPClient | None = None
_pipeline = None  # compiled LangGraph app


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_client, _pipeline
    setup_tracing()
    logger.info("Initialising ADAPT-AI pipeline…")
    _mcp_client = build_mcp_client()
    _pipeline = build_graph(_mcp_client)
    logger.info("Pipeline ready")
    yield
    logger.info("ADAPT-AI shutting down")


app = FastAPI(title="ADAPT-AI", version="2.0.0", lifespan=lifespan)

# Serve ui/ as static files (CSS, JS assets if any)
if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR)), name="ui")


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    subject_id: Optional[str] = None   # domain entity id (patient/case/account)
    session_id: Optional[str] = None
    domain: str = "healthcare"


class QueryResponse(BaseModel):
    status: str
    content: str
    agent_statuses: dict
    metadata: dict


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(_UI_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline": "ready" if _pipeline else "initialising"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    session_id = request.session_id or str(uuid.uuid4())
    t0 = time.perf_counter()

    # Restore previous session context
    session = SessionManager.get()
    ctx = await session.get_context(session_id)

    initial_state = {
        "query": request.query,
        "subject_id": request.subject_id,
        "domain": request.domain,
        "session_id": session_id,
        "use_rat": False,
        "retrieved_context": "",
        "primary_response": "",
        "compliance_result": {},
        "quality_result": {},
        "final_response": "",
        "revision_count": 0,
        "revision_feedback": "",
        "agent_statuses": {},
        "llm_usage": None,
        "error": None,
    }

    try:
        result = await _pipeline.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}},
        )
    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round(time.perf_counter() - t0, 3)

    # Handle compliance rejection (pipeline ends early → no final_response)
    if not result.get("final_response") and not result.get("compliance_result", {}).get("passed", True):
        await session.append_message(session_id, "user", request.query)
        compliance = result.get("compliance_result", {})
        issues = compliance.get("issues", [])
        detail_lines = [
            f"- [{i.get('severity', 'n/a').upper()}] {i.get('description', '')}"
            + (f" (matched: \"{i['matched_text']}\")" if i.get("matched_text") else "")
            for i in issues
        ]
        content = "**Response withheld — compliance check failed.**\n\n" + (
            "\n".join(detail_lines) if detail_lines else "No specific issues reported."
        )
        return QueryResponse(
            status="rejected",
            content=content,
            agent_statuses=result.get("agent_statuses", {}),
            metadata={
                "session_id": session_id,
                "response_time": elapsed,
                "compliance_issues": issues,
            },
        )

    final = result.get("final_response", result.get("primary_response", ""))

    # Persist to session
    await session.append_message(session_id, "user", request.query)
    await session.append_message(session_id, "assistant", final)
    await session.save_context(session_id, {"last_query": request.query})

    return QueryResponse(
        status="success",
        content=final,
        agent_statuses=result.get("agent_statuses", {}),
        metadata={
            "session_id": session_id,
            "subject_id": request.subject_id,
            "response_time": elapsed,
            "use_rat": result.get("use_rat", False),
            "revision_count": result.get("revision_count", 0),
        },
    )


@app.get("/patients")
async def list_patients():
    ph = PatientHandler()
    return ph.list_patients()


@app.get("/session/{session_id}/history")
async def session_history(session_id: str):
    session = SessionManager.get()
    history = await session.get_conversation_history(session_id)
    return {"session_id": session_id, "messages": history}
