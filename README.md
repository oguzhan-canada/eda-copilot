# EDA Copilot

**LLM-Powered Knowledge Graph for Electronic Design Automation**

A GraphRAG copilot for chip design engineers that combines a knowledge graph, vector retrieval, and LoRA fine-tuning to answer EDA questions with version-aware, structured reasoning.

🔗 **[Live Dashboard](https://oguzhan-canada.github.io/eda-copilot/)** · 📄 **[Research Paper](paper/research_paper.pdf)** · 📊 **[PROGRESS.md](PROGRESS.md)**

---

## Key Results

| System | Answer Quality | Graph Hit % |
|--------|---------------|-------------|
| **Full GraphRAG** | **0.482** | **67.5%** |
| LoRA-only | 0.431 | 0.0% |
| Vector-only RAG | 0.202 | 0.0% |
| Direct LLM | 0.022 | 0.0% |

- **21.9×** improvement over bare LLM
- **139%** over vector-only search
- **67.5%** graph-grounded answers
- **$240** total build cost (10% of $2,685 original budget)

## System Architecture

Six-component pipeline built over 16 weeks:

1. **Corpus** — 14,294 files, 252M tokens from 6 open-source EDA sources
2. **Synthetic Q&A** — 13,024 pairs across 12 violation families, judge threshold ≥ 0.90
3. **Knowledge Graph** — 18,037 nodes, 16,530 relationships in Neo4j, p50 2-hop latency 45ms
4. **Vector Index** — 8,888 priority chunks via voyage-code-2 (1,536 dim) in Weaviate
5. **Fusion Retrieval** — Parallel Neo4j + Weaviate search with ms-marco cross-encoder reranking
6. **LoRA Adaptation** — Mistral-7B QLoRA 4-bit, 12.5× perplexity improvement, $9.09 training cost

## EDABench

A 120-item zero-contamination benchmark for EDA copilot evaluation:
- 5 task categories: Error Diagnosis, RTL Q&A, Constraint Generation, DRC Rule Lookup, Cross-Tool Knowledge
- 7 seed bugs from real OpenROAD runs (ED-001–005, ML-001–002)
- KG-grounded ground truth for 33 items

## Tech Stack

| Component | Technology |
|-----------|------------|
| Knowledge Graph | Neo4j Aura |
| Vector Store | Weaviate Cloud |
| Embeddings | voyage-code-2 (1,536 dim) |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| Fine-tuned Model | Mistral-7B-Instruct-v0.3 + QLoRA |
| Synthesis | Claude Sonnet API |
| API | FastAPI (Python) |

## Related Projects

- **[ML-Driven PPA Optimization](https://oguzhan-canada.github.io/instrumented-ml-ppa/findings.html)** — The prior MLCAD 2026 project that discovered the ORFS v3.0 → 26Q1 version divergence anchoring this system
- **[AI Inference Cost Optimizer](https://oguzhan-canada.github.io/ai-inference-cost-optimizer/)** — FinOps decision tool using the same cost engineering principles

## Author

**Oguzhan Tekin** — Machine Learning and Artificial Intelligence Researcher · Toronto

- GitHub: [oguzhan-canada](https://github.com/oguzhan-canada)
- Email: oguzhantekin@gmail.com
