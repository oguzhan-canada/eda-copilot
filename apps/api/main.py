"""
FastAPI endpoint for the EDA Copilot GraphRAG system.

Single endpoint:
  POST /query — accepts a question, returns answer + citations

Usage:
    uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic

from apps.api.services.fusion_retriever import FusionRetriever
from apps.api.services.synthesizer import Synthesizer


# ── Request / Response models ────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    search_mode: str = "hybrid"  # "hybrid", "dense", "sparse"
    synthesize: bool = True  # False = retrieval only (no Claude call)


class Citation(BaseModel):
    source_file: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    task_category: str
    confidence: str
    entities_found: list[str]
    graph_facts_count: int
    chunks_count: int
    usage: dict
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    weaviate: str
    kg_nodes: Optional[int] = None


# ── App lifecycle ────────────────────────────────────────────────────────

retriever: Optional[FusionRetriever] = None
synthesizer: Optional[Synthesizer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, synthesizer
    retriever = FusionRetriever()
    synthesizer = Synthesizer()
    yield
    if retriever:
        retriever.close()


app = FastAPI(
    title="EDA Copilot API",
    description="GraphRAG-powered EDA assistant combining Neo4j knowledge graph and Weaviate vector search",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer an EDA question using graph + vector retrieval + Claude synthesis."""
    if not retriever or not synthesizer:
        raise HTTPException(status_code=503, detail="Services not initialized")

    start = time.time()

    # Step 1: Retrieve
    result = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        search_mode=request.search_mode,
    )

    # Step 2: Synthesize (optional)
    if request.synthesize:
        answer = synthesizer.synthesize(request.query, result)
    else:
        answer = {
            "answer": "Retrieval only — synthesis disabled",
            "citations": [
                c.get("source_file", "") for c in result.get("chunks", [])
            ],
            "confidence": "n/a",
            "usage": {},
        }

    latency = (time.time() - start) * 1000

    return QueryResponse(
        answer=answer.get("answer", ""),
        citations=answer.get("citations", []),
        task_category=result.get("task_category", "unknown"),
        confidence=answer.get("confidence", "low"),
        entities_found=result.get("entities_found", []),
        graph_facts_count=len(result.get("graph_facts", [])),
        chunks_count=len(result.get("chunks", [])),
        usage=answer.get("usage", {}),
        latency_ms=round(latency, 1),
    )


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream an EDA answer — words appear as Claude generates them."""
    if not retriever or not synthesizer:
        raise HTTPException(status_code=503, detail="Services not initialized")

    result = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        search_mode=request.search_mode,
    )

    context = synthesizer._format_context(result)
    max_chars = 6000 * 4
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[Context truncated for length]"

    user_message = f"## Context\n\n{context}\n\n## Query\n\n{request.query}\n\nRespond with a clear, detailed answer. Do NOT use JSON format — write a natural language answer. Cite source files inline."

    import json as _json
    metadata = {
        "task_category": result.get("task_category", "unknown"),
        "graph_facts_count": len(result.get("graph_facts", [])),
        "chunks_count": len(result.get("chunks", [])),
        "entities_found": result.get("entities_found", []),
        "citations": [c.get("source_file", "") for c in result.get("chunks", []) if c.get("source_file")],
    }

    async def generate():
        yield _json.dumps(metadata) + "\n"
        client = anthropic.Anthropic(api_key=synthesizer.api_key)
        with client.messages.stream(
            model=synthesizer.model,
            max_tokens=1024,
            system="You are an EDA copilot. Answer using ONLY the provided context. Cite source files inline. For errors: Root Cause, Evidence, Fix, Caveats. State version info explicitly.",
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check connectivity to Neo4j and Weaviate."""
    neo4j_status = "unknown"
    weaviate_status = "unknown"
    kg_nodes = None

    if retriever:
        try:
            with retriever.graph.driver.session() as s:
                count = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            neo4j_status = "connected"
            kg_nodes = count
        except Exception as e:
            neo4j_status = f"error: {str(e)[:80]}"

        try:
            collection = retriever.weaviate_engine.collection
            obj_count = collection.aggregate.over_all(total_count=True).total_count
            weaviate_status = f"connected ({obj_count} objects)"
        except Exception as e:
            weaviate_status = f"error: {str(e)[:80]}"

    return HealthResponse(
        status="ok" if "connected" in neo4j_status and "connected" in weaviate_status else "degraded",
        neo4j=neo4j_status,
        weaviate=weaviate_status,
        kg_nodes=kg_nodes,
    )
