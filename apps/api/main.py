"""
FastAPI endpoint for the EDA Copilot GraphRAG system.

Single endpoint:
  POST /query — accepts a question, returns answer + citations

Usage:
    uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import anthropic

from apps.api.services.fusion_retriever import FusionRetriever
from apps.api.services.synthesizer import Synthesizer


# ── Request / Response models ────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    search_mode: str = "hybrid"
    synthesize: bool = True

    @field_validator("query")
    @classmethod
    def cap_query_length(cls, v: str) -> str:
        if len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        if len(v) > 500:
            raise ValueError("Query too long — max 500 characters")
        return v.strip()

    @field_validator("top_k")
    @classmethod
    def cap_top_k(cls, v: int) -> int:
        return max(1, min(v, 10))


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


# ── Rate limiting ─────────────────────────────────────────────────────────

def get_real_ip(request: Request) -> str:
    """Get real client IP from nginx X-Real-IP / X-Forwarded-For headers."""
    return (
        request.headers.get("X-Real-IP")
        or (request.headers.get("X-Forwarded-For", "").split(",")[0].strip())
        or request.client.host
    )

limiter = Limiter(key_func=get_real_ip)

logger = logging.getLogger("eda-copilot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


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
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


# ── Endpoints ────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest):
    """Answer an EDA question using graph + vector retrieval + Claude synthesis."""
    if not retriever or not synthesizer:
        raise HTTPException(status_code=503, detail="Services not initialized")

    start = time.time()

    result = retriever.retrieve(
        query=body.query,
        top_k=body.top_k,
        search_mode=body.search_mode,
    )

    if body.synthesize:
        answer = synthesizer.synthesize(body.query, result)
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
    usage = answer.get("usage", {})
    logger.info(
        "QUERY | endpoint=/query | tokens_in=%s | tokens_out=%s | latency=%dms | category=%s | ip=%s",
        usage.get("input_tokens", "?"), usage.get("output_tokens", "?"),
        int(latency), result.get("task_category", "unknown"),
        get_real_ip(request),
    )

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
@limiter.limit("10/minute")
async def query_stream(request: Request, body: QueryRequest):
    """Stream an EDA answer via SSE — metadata first, then tokens, then done."""
    import json as json_lib

    if not retriever or not synthesizer:
        raise HTTPException(status_code=503, detail="Services not initialized")

    t0 = time.time()

    result = retriever.retrieve(
        query=body.query,
        top_k=body.top_k,
        search_mode=body.search_mode,
    )

    context = synthesizer._format_context(result)
    max_chars = 6000 * 4
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[Context truncated for length]"

    user_message = f"## Context\n\n{context}\n\n## Query\n\n{body.query}\n\nRespond with a clear, detailed answer. Do NOT use JSON format — write a natural language answer. Cite source files inline."

    async def generate():
        # First chunk: metadata so UI can show debug strip immediately
        meta = {
            "type": "meta",
            "task_category": result.get("task_category", "unknown"),
            "graph_fact_count": len(result.get("graph_facts", [])),
            "chunk_count": len(result.get("chunks", [])),
            "citations": [
                c.get("source_file", "")
                for c in result.get("chunks", [])
                if c.get("source_file")
            ],
        }
        yield f"data: {json_lib.dumps(meta)}\n\n"

        # Stream Claude answer token by token
        client = anthropic.Anthropic(api_key=synthesizer.api_key)
        input_tokens = 0
        output_tokens = 0
        with client.messages.stream(
            model=synthesizer.model,
            max_tokens=1024,
            system="You are an EDA copilot. Answer using ONLY the provided context. Cite source files inline. For errors: Root Cause, Evidence, Fix, Caveats. State version info explicitly.",
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                chunk = {"type": "token", "text": text}
                yield f"data: {json_lib.dumps(chunk)}\n\n"
            final = stream.get_final_message()
            input_tokens = final.usage.input_tokens
            output_tokens = final.usage.output_tokens

        # Final chunk: latency + token usage
        latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            "QUERY | endpoint=/query/stream | latency=%dms | tokens_in=%d | tokens_out=%d | category=%s | graph_facts=%d | chunks=%d | ip=%s",
            latency_ms, input_tokens, output_tokens,
            result.get("task_category", "unknown"),
            len(result.get("graph_facts", [])), len(result.get("chunks", [])),
            get_real_ip(request),
        )
        done = {"type": "done", "latency_ms": latency_ms, "input_tokens": input_tokens, "output_tokens": output_tokens}
        yield f"data: {json_lib.dumps(done)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
