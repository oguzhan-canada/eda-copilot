# LLM-Powered Knowledge Graphs for EDA

**A version-aware EDA copilot combining corpus ingestion, Neo4j knowledge graphs, hybrid GraphRAG retrieval, QLoRA domain adaptation, and EDABench evaluation.**

Oguzhan Tekin · Machine Learning and AI Researcher · Toronto, Canada

🌐 [Live Dashboard](https://oguzhan-canada.github.io/eda-copilot/) · 📄 [Research Paper](docs/paper/research_paper.pdf) · 📊 [Progress Trail](PROGRESS.md) · 🏗️ [Architecture Guide](https://oguzhan-canada.github.io/eda-copilot/architecture-guide.html) · 💰 [Cost Optimization Paper](https://oguzhan-canada.github.io/eda-copilot/cost-optimization.html)

Chip designers spend enormous amounts of time searching through tool logs, version changelogs, and community forums to diagnose failures they have already seen before. This project builds a smarter assistant — one that doesn't just search text, but actually understands the relationships between errors, fixes, tool versions, and design rules. By combining a Neo4j knowledge graph with vector search and a large language model, the system answers chip design questions with 21.9× higher quality than a standalone LLM, while citing exactly which documents and graph relationships informed each answer — all built for $240 on open-source tools.

---

## Quick Start → Week 1

```bash
# 1. Clone and set up
cd ~/projects
git clone <this-repo> llm-eda-kg
cd llm-eda-kg

# 2. Verify config
python3 -c "import yaml; cfg=yaml.safe_load(open('configs/base.yaml')); print('OK:', cfg['project_name'], '/', len(cfg['designs']), 'designs')"

# 3. Set up AWS (see infra/aws/terraform/terraform.tfvars.example)
cp infra/aws/terraform/terraform.tfvars.example infra/aws/terraform/terraform.tfvars
# Edit terraform.tfvars with your values
cd infra/aws/terraform && terraform init && terraform plan
```

---

## Overview

An end-to-end system for building knowledge-driven EDA assistants that reason over chip design artifacts using structured knowledge graphs rather than flat text retrieval. Built on empirical foundations from the [Instrumented ML-Driven PPA Optimization](https://github.com/oguzhan-canada/instrumented-ml-ppa) project.

**Key capabilities:**
- **4-tier corpus pipeline** — structured code, PDK artifacts, tool docs, synthetic Q&A
- **Version-aware knowledge graph** — Neo4j ontology with 10 node types, 10 relation types, provenance tracking
- **Hybrid GraphRAG retrieval** — dense + BM25 + graph subgraph expansion for multi-hop reasoning
- **QLoRA domain adaptation** — Llama-3-8B fine-tuned for Verilog/SDC generation
- **EDABench** — 900-sample benchmark with 6 verified seed samples from real ORFS experiments

## Repository Structure

```
├── apps/                    # API server and UI
│   ├── api/                 # FastAPI backend
│   │   ├── routers/         # Endpoint definitions
│   │   └── services/        # Query router, GraphRAG, citations
│   └── ui/                  # Gradio pilot interface
│
├── configs/                 # Central configuration
│   ├── base.yaml            # Master config (paths, designs, services)
│   ├── corpus_sources.yaml  # Tier 1-3 ingestion targets
│   └── graph_schema.cypher  # Neo4j ontology + seed triples
│
├── data_contracts/          # Schema definitions
│   └── manifest_schema.yaml # Artifact, ORFS run, and triple manifests
│
├── infra/                   # Infrastructure-as-Code
│   ├── aws/terraform/       # EC2 spot + S3 (from MLCAD)
│   ├── aws/docker/          # Container definitions
│   ├── aws/scripts/         # Bootstrap and sync
│   └── compose/             # Local dev docker-compose
│
├── pipeline/                # Data processing pipeline
│   ├── collect/             # Corpus download and staging
│   ├── synth_qa/            # Synthetic Q&A generation
│   ├── orfs/                # ORFS execution (from MLCAD)
│   ├── parse/               # Log/report parsing (from MLCAD)
│   ├── graph/               # Triple extraction and KG loading
│   ├── retrieve/            # Chunking, embedding, hybrid search
│   ├── finetune/            # QLoRA training pipeline
│   └── eval/                # EDABench construction and scoring
│
├── tests/                   # Test suite
├── requirements/            # Split dependency files
├── experiments/legacy/      # MLCAD reference scripts
├── data/                    # Data artifacts (gitignored, synced via S3)
├── models/                  # Trained models (gitignored)
├── results/                 # Evaluation outputs (gitignored)
└── vendor/                  # MLCAD source repo (gitignored)
```

## Phases

| Phase | Weeks | Focus | Key Deliverable |
|-------|-------|-------|-----------------|
| **1** | 1–4 | Corpus & Data Pipeline | 252M-token staged corpus, 10K synthetic Q&A |
| **2** | 5–9 | Knowledge Graph | Neo4j ontology, 18,035 nodes, version-aware edges |
| **3** | 10–14 | RAG + Fine-tuning | Hybrid GraphRAG, QLoRA adapter, API server |
| **4** | 15–16 | Evaluation + Deployment | 120-sample EDABench, live production system |

## Live System

The system is deployed and publicly accessible:

- **Dashboard**: [oguzhan-canada.github.io/eda-copilot](https://oguzhan-canada.github.io/eda-copilot/)
- **API**: HTTPS endpoint with SSE streaming (~2–3s perceived latency)
- **Architecture Guide**: [Implementation reference](https://oguzhan-canada.github.io/eda-copilot/architecture-guide.html)

### Production Stack

| Component | Technology | Status |
|---|---|---|
| Knowledge Graph | Neo4j Aura (18,035 nodes) | ✅ Connected |
| Vector Store | Weaviate Cloud (8,958 objects) | ✅ Connected |
| LLM Synthesis | Claude Sonnet via Anthropic API | ✅ Streaming |
| API Server | FastAPI + uvicorn + systemd | ✅ Running |
| Reverse Proxy | nginx + Let's Encrypt SSL | ✅ HTTPS |
| Frontend | GitHub Pages (single-file HTML) | ✅ Live |
| CI | GitHub Actions keepalive ping | ✅ Active |

### Key Results

| Metric | Value |
|---|---|
| Answer quality vs standalone LLM | **21.9× improvement** |
| Mean answer quality (EDABench) | 0.482 |
| Graph hit rate | 67.5% |
| Per-query cost | ~$0.01 |
| Total build cost | **$240** (91% under $2,685 original budget) |

## Reused Assets from MLCAD Project

| Asset | New Location | Role |
|-------|-------------|------|
| `run_openroad.py` | `pipeline/orfs/` | ORFS parallel execution |
| `extract_timing.py` | `pipeline/parse/` | STA report parsing |
| `parse_logs.py` | `pipeline/parse/` | Power/area extraction |
| `fix_def_instances.py` | `pipeline/parse/` | DEF normalization |
| `terraform/` | `infra/aws/terraform/` | AWS infrastructure |
| 4 ORFS bugs | `data/edabench/seeds/` | EDABench gold standards |

## Budget

| Item | Budgeted | Actual |
|------|----------|--------|
| Total build (16 weeks) | ~$2,685 | **$240** |
| Monthly recurring (VPS) | — | ~$3/mo |
| Per-query API cost | — | ~$0.01 |

## License

MIT
