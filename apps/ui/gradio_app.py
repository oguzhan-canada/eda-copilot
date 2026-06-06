"""
Gradio UI for the EDA Knowledge Graph Copilot.

Connects to the FastAPI backend (apps/api/main.py) or runs the
pipeline directly when the API server is not available.

Usage:
    # With FastAPI backend running:
    python -m apps.ui.gradio_app

    # Standalone (direct pipeline, no API server needed):
    python -m apps.ui.gradio_app --standalone
"""

import argparse
import json
import time
from typing import Optional

import gradio as gr

# ---------------------------------------------------------------------------
# Backend connector — API mode or standalone
# ---------------------------------------------------------------------------

_retriever = None
_synthesizer = None


def _init_standalone():
    """Initialize pipeline components for standalone mode."""
    global _retriever, _synthesizer
    if _retriever is None:
        from apps.api.services.fusion_retriever import FusionRetriever
        from apps.api.services.synthesizer import Synthesizer

        _retriever = FusionRetriever()
        _synthesizer = Synthesizer()


def query_via_api(question: str, mode: str, top_k: int) -> dict:
    """Call the FastAPI backend."""
    import requests

    try:
        resp = requests.post(
            "http://localhost:8000/query",
            json={"query": question, "search_mode": mode.lower(), "top_k": top_k},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"answer": f"API error: {e}", "citations": [], "task_category": "unknown",
                "confidence": "error", "entities_found": [], "graph_facts_count": 0,
                "chunks_count": 0, "usage": {}, "latency_ms": 0}


def query_standalone(question: str, mode: str, top_k: int) -> dict:
    """Run the pipeline directly without the API server."""
    _init_standalone()
    start = time.time()
    result = _retriever.retrieve(query=question, top_k=top_k, search_mode=mode.lower())
    answer = _synthesizer.synthesize(question, result)
    latency = (time.time() - start) * 1000
    return {
        "answer": answer.get("answer", ""),
        "citations": answer.get("citations", []),
        "task_category": result.get("task_category", "unknown"),
        "confidence": answer.get("confidence", "low"),
        "entities_found": result.get("entities_found", []),
        "graph_facts_count": len(result.get("graph_facts", [])),
        "chunks_count": len(result.get("chunks", [])),
        "usage": answer.get("usage", {}),
        "latency_ms": round(latency, 1),
    }


# ---------------------------------------------------------------------------
# Gradio handler
# ---------------------------------------------------------------------------

_use_standalone = False


def ask_copilot(question: str, mode: str, top_k: int):
    """Main handler: query the system and return formatted output."""
    if not question.strip():
        return "", "", ""

    if _use_standalone:
        data = query_standalone(question, mode, top_k)
    else:
        data = query_via_api(question, mode, top_k)

    answer = data.get("answer", "No answer generated.")

    citations = data.get("citations", [])
    if citations:
        citations_text = "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(citations))
    else:
        citations_text = "  (no sources cited)"

    entities = data.get("entities_found", [])
    debug_lines = [
        f"Category:    {data.get('task_category', '?')}",
        f"Confidence:  {data.get('confidence', '?')}",
        f"Graph facts: {data.get('graph_facts_count', 0)}",
        f"Chunks:      {data.get('chunks_count', 0)}",
        f"Entities:    {', '.join(entities) if entities else 'none'}",
        f"Latency:     {data.get('latency_ms', 0):.0f} ms",
    ]
    usage = data.get("usage", {})
    if usage:
        debug_lines.append(f"Model:       {usage.get('model', '?')}")
        debug_lines.append(f"Tokens:      {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out")

    return answer, citations_text, "\n".join(debug_lines)


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="EDA Copilot",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "## EDA Knowledge Graph Copilot\n"
            "Version-aware assistant for OpenROAD, ORFS, Yosys, OpenSTA, and SKY130/ASAP7 PDKs.\n"
            "Powered by GraphRAG: Neo4j knowledge graph + Weaviate vector search + Claude synthesis."
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_box = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g. Why does JPEG timing fail after upgrading to ORFS 26Q1?",
                    lines=3,
                )
            with gr.Column(scale=1):
                mode = gr.Radio(
                    ["hybrid", "dense", "sparse"],
                    value="hybrid",
                    label="Retrieval mode",
                )
                top_k = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Top-K chunks",
                )
                submit_btn = gr.Button("Ask", variant="primary", size="lg")

        answer_box = gr.Textbox(label="Answer", lines=10, interactive=False)

        with gr.Row():
            citations_box = gr.Textbox(label="Sources", lines=5, interactive=False)
            debug_box = gr.Textbox(label="Retrieval info", lines=5, interactive=False)

        gr.Examples(
            examples=[
                ["Why does JPEG timing fail after upgrading from ORFS v3.0 to 26Q1?", "hybrid", 5],
                ["OpenROAD crashes with SIGSEGV on ibex in global routing", "hybrid", 5],
                ["How do I write an SDC constraint for a 500MHz clock in nanoseconds?", "sparse", 5],
                ["What is the metal2 spacing rule in SKY130?", "dense", 5],
                ["SDC time unit mismatch producing implausible WNS values", "hybrid", 5],
            ],
            inputs=[query_box, mode, top_k],
        )

        submit_btn.click(
            ask_copilot,
            inputs=[query_box, mode, top_k],
            outputs=[answer_box, citations_box, debug_box],
        )
        query_box.submit(
            ask_copilot,
            inputs=[query_box, mode, top_k],
            outputs=[answer_box, citations_box, debug_box],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA Copilot Gradio UI")
    parser.add_argument("--standalone", action="store_true",
                        help="Run pipeline directly instead of calling FastAPI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    _use_standalone = args.standalone
    if _use_standalone:
        print("Running in standalone mode (no API server needed)")

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)
