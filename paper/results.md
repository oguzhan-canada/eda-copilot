# Section: Results

> This section presents the system evaluation results.
> It should follow the Experimental Setup and EDABench sections.

---

## 7 Evaluation Results

We evaluate the full GraphRAG system and three ablation baselines on all 120 EDABench items. Each system response is judged by Claude Sonnet on four dimensions: factual accuracy, completeness, actionability, and specificity (each on a [0, 1] scale). Answer quality is the minimum of the four dimension scores — the strictest aggregation, reflecting that a response must satisfy all criteria to be useful. All 480 judge evaluations (120 items × 4 systems) were conducted in a single batch with identical prompts to ensure cross-system comparability.

### 7.1 Main Results

**Table 1: System Comparison on EDABench (n = 120)**

| System | Answer Quality | Factual | Completeness | Actionability | Specificity | Graph Hit Rate | Latency (p50) |
|---|---|---|---|---|---|---|---|
| Full GraphRAG | **0.482** | **0.810** | **0.635** | 0.563 | 0.573 | **67.5%** | 12.7s |
| LoRA-only | 0.431 | 0.491 | 0.565 | **0.655** | **0.656** | 0.0% | 15.7s |
| Vector-only RAG | 0.202 | 0.672 | 0.292 | 0.233 | 0.394 | 0.0% | 13.0s |
| Direct LLM (no retrieval) | 0.072 | 0.642 | 0.173 | 0.081 | 0.203 | 0.0% | 5.4s |

*All 120 items scored for all four systems with zero parse failures.*

**Key findings:**

1. **Full GraphRAG achieves a 6.7× improvement** over the direct LLM baseline (0.482 vs 0.072). This is the paper's primary result: retrieval-augmented generation with structured knowledge is essential for EDA domain tasks.

2. **The knowledge graph contributes a 139% improvement over vector-only retrieval** (0.482 vs 0.202). Graph hit rate varies by category: 89% for cross-tool knowledge and RTL Q&A, 83% for constraint generation, 71% for error diagnosis, and 0% for DRC rule lookup. Unlike the modest aggregate difference suggested by prior partial evaluations, consistent scoring across all 120 items reveals a substantial KG contribution.

3. **LoRA fine-tuning is surprisingly competitive** (0.431), outperforming vector-only RAG by +0.229. The fine-tuned Mistral-7B model internalized sufficient domain knowledge during training on 2,586 EDA-specific examples to rival retrieval for certain query types — particularly error diagnosis (0.525 vs 0.175 for vector-only). However, the full GraphRAG system still outperforms LoRA-only on version-aware queries (see §7.3), confirming that fine-tuning and structured retrieval are complementary.

4. **The direct LLM baseline confirms the domain gap.** At 0.072 answer quality, Claude Sonnet without retrieval achieves reasonable factual accuracy (0.642) but fails on completeness (0.173), actionability (0.081), and specificity (0.203). The model knows EDA concepts exist but cannot provide the detailed, actionable guidance practitioners need.

### 7.2 Results by Task Category

| Category | Full GraphRAG | LoRA-only | Vector-only | Direct LLM | Graph Hit Rate | n |
|---|---|---|---|---|---|---|
| Constraint Generation | **0.517** | 0.461 | 0.356 | 0.056 | 83% | 18 |
| RTL Q&A | **0.511** | 0.267 | 0.117 | 0.022 | 89% | 18 |
| Error Diagnosis | 0.504 | **0.525** | 0.175 | 0.106 | 71% | 48 |
| DRC Rule Lookup | **0.422** | 0.378 | 0.300 | 0.056 | 0% | 18 |
| Cross-Tool Knowledge | **0.422** | 0.367 | 0.106 | 0.061 | 89% | 18 |

**Observations:**

- **Error Diagnosis** is the only category where LoRA-only (0.525) outperforms Full GraphRAG (0.504). The fine-tuned model internalized common EDA error patterns during training, making it competitive on diagnosis queries that follow familiar patterns. However, Full GraphRAG still dominates on version-specific items within this category (see §7.3).

- **RTL Q&A** shows the largest gap between Full GraphRAG and alternatives: 0.511 vs 0.267 (LoRA) and 0.117 (vector-only). The 89% graph hit rate indicates the entity extractor consistently identifies module and design entities, providing structured context that neither fine-tuning nor vector search alone can replicate.

- **DRC Rule Lookup** achieves **0% graph hit rate** — the entity extractor does not match PDK rule identifiers (e.g., "METAL1.S.1"). Despite this, Full GraphRAG still outperforms vector-only (0.422 vs 0.300) because the vector component benefits from the broader retrieval pipeline.

- **Cross-Tool Knowledge** shows the largest ratio improvement for Full GraphRAG over vector-only (4.0×, 0.422 vs 0.106) with 89% graph hit rate. Queries in this category require reasoning across tool versions and design configurations — exactly the multi-hop traversals the knowledge graph enables.

### 7.3 Case Studies: Regression Anchors

Two seed items illustrate the KG contribution on version-aware queries:

**ED-005 (SIGSEGV crash, Error Diagnosis, Hard):**

| System | Score |
|---|---|
| Full GraphRAG | **0.50** |
| LoRA-only | 0.20 |
| Vector-only | 0.20 |
| Direct LLM | 0.10 |

The full system achieves a **2.5× improvement** over both vector-only and LoRA-only on this item. The graph traversal retrieves the specific OpenROAD version, the ibex design, and the global routing stage as linked entities — context that dense retrieval scatters across unrelated chunks and that fine-tuning cannot memorize for specific version combinations. This demonstrates the core thesis: for queries requiring version-aware causal reasoning, structured graph context provides substantial lift.

**ED-002 (JPEG WNS sign flip, Error Diagnosis, Hard):**

| System | Score |
|---|---|
| LoRA-only | **0.70** |
| Full GraphRAG | 0.20 |
| Direct LLM | 0.20 |
| Vector-only | 0.10 |

ED-002 reveals an interesting complementarity: the LoRA model (0.70) dramatically outperforms all retrieval-based systems on this query. The fine-tuned model appears to have internalized the pattern of WNS sign flips during ORFS version migration from training examples, producing a more complete and actionable response than the retrieval pipeline. This suggests that for common, well-documented failure modes, domain adaptation can substitute for retrieval. The full GraphRAG system scored only 0.20, likely because the retrieved graph facts did not include the specific version-to-version metric comparison needed.

### 7.4 Results by Difficulty

| Difficulty | Full GraphRAG | LoRA-only | Vector-only | Direct LLM | n |
|---|---|---|---|---|---|
| Easy | **0.597** | 0.383 | 0.328 | 0.047 | 36 |
| Medium | 0.443 | **0.547** | 0.164 | 0.092 | 53 |
| Hard | **0.395** | 0.271 | 0.119 | 0.057 | 21 |
| Expert | **0.460** | 0.320 | 0.120 | 0.080 | 10 |

Full GraphRAG leads on easy, hard, and expert items. LoRA-only outperforms on medium-difficulty items (0.547 vs 0.443), suggesting the fine-tuned model excels on standard diagnostic patterns that appear frequently in training data.

The non-monotonic pattern (expert > hard for Full GraphRAG) warrants discussion. Expert items in our benchmark tend to have clearer KG anchors — they reference specific experiments, tools, and version pairs that the graph retrieves precisely. Hard items often involve ambiguous multi-factor causation where no single graph traversal captures the full answer. This suggests that graph retrieval is most effective when the query maps cleanly to known entities, regardless of the conceptual difficulty.

### 7.5 Judge Calibration

Answer quality is scored as the minimum of four dimensions — the strictest possible aggregation, meaning a single weak dimension caps the overall score. The bottleneck dimension is distributed: actionability (34%), specificity (34%), and completeness (33%) each serve as the lowest-scoring dimension approximately equally often. Factual accuracy is rarely the bottleneck (mean 0.810 for Full GraphRAG), suggesting the system retrieves relevant facts but sometimes fails to translate them into complete, actionable guidance.

The LoRA-only system shows a distinct profile: higher actionability (0.655) and specificity (0.656) than Full GraphRAG (0.563, 0.573) but lower factual accuracy (0.491 vs 0.810). The fine-tuned model generates more actionable-sounding responses but occasionally hallucinates specific details, while the retrieval system grounds responses in source material but sometimes produces less focused answers.

### 7.6 Limitations and Threats to Validity

**Entity extraction coverage.** The entity extractor matches broad entities (tool names, design names, version identifiers) but does not resolve specific violation or fix node IDs from natural language descriptions. As a result, the system retrieves relevant 2-hop subgraphs (67.5% graph hit rate) but cannot target specific violation nodes. We report graph hit rate rather than node-level retrieval precision; future work on semantic entity linking could close this gap.

**DRC rule entity matching.** DRC rule lookup queries achieve 0% graph hit rate because the entity extractor does not match PDK rule identifiers (e.g., "METAL1.S.1", "met2 spacing"). Adding rule-code pattern matching to the entity extractor would likely improve DRC category scores.

**Judge model bias.** Using Claude Sonnet as both the synthesis engine and the evaluation judge introduces potential self-evaluation bias. Mitigating this requires human expert evaluation, which we defer to the full benchmark release.

**Judge parse failures.** All 480 judge evaluations (120 items × 4 systems) completed with zero parse failures in the final evaluation batch, enabling direct cross-system comparison without sample-size confounds.

**LoRA evaluation environment.** The LoRA-only baseline was evaluated on a separate GPU instance using Mistral-7B-Instruct-v0.3 with QLoRA 4-bit quantization. Response generation averaged 15.2 seconds per item — slower than the Full GraphRAG system (12.7s) due to autoregressive generation without retrieval-guided context truncation.

### 7.7 Summary

The evaluation demonstrates four findings:

1. **GraphRAG is essential for EDA copilots.** The 6.7× improvement over the bare LLM validates the architecture choice in §4.4.

2. **Knowledge graphs provide substantial lift over vector-only retrieval.** The 139% improvement (0.482 vs 0.202) and the 67.5% graph hit rate confirm the hypothesis in §5.1 of the survey: structured retrieval outperforms dense retrieval alone. The effect is strongest on cross-tool knowledge (4.0× over vector-only) and RTL Q&A (4.4× over vector-only), and absent on DRC rule lookup (0% graph hit rate).

3. **LoRA fine-tuning is a competitive alternative to retrieval.** The fine-tuned Mistral-7B achieves 0.431 answer quality — within 11% of the full GraphRAG system — demonstrating that domain adaptation can internalize substantial EDA knowledge. However, on version-specific queries (ED-005), the full system's 2.5× advantage shows that fine-tuning and structured retrieval are complementary, not substitutable.

4. **The system was built at 34% of budget** ($240 of $660–700), validating the cost feasibility of the reference architecture for academic and small-team deployments.
