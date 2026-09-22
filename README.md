# Self-Improving Multi-Agent RAG with Hallucination Detection

This repository implements a **Self-Improving Multi-Agent RAG with Hallucination Detection** system, divided into two distinct development phases:

- **Part 1 (Current Focus)**: **Hybrid RAG Baseline & Evaluation Benchmark**
- **Part 2 (Future Phase)**: **Multi-Agent Orchestration & Self-Improving Hallucination Recovery Layer**

---

## 🎯 Part 1: RAG System Specification

### 1. Technology Stack
| Component | Technology / Model | Rationale |
| :--- | :--- | :--- |
| **Language & Framework** | Python 3.10+ / LangChain | Core orchestration and retrieval chain |
| **LLM Inference** | Groq (`llama-3.3-70b-versatile` / `mixtral-8x7b-32768`) | Ultra-fast inference with strict grounding |
| **Vector Database** | Qdrant | Dense vector search, hybrid retrieval, payload filtering |
| **Dense Embeddings** | `BAAI/bge-m3` | State-of-the-art dense semantic representations |
| **Sparse Retrieval** | BM25 | Exact keyword, entity, name, and terminology matching |
| **Fusion Algorithm** | Reciprocal Rank Fusion (RRF) | Merging dense semantic and sparse lexical ranks |
| **Reranker** | Cross-Encoder `BAAI/bge-reranker-v2-m3` | Joint query-passage deep scoring |
| **Benchmark Dataset** | 2WikiMultiHopQA | Multi-hop reasoning with ground-truth supporting facts |
| **Evaluation Framework** | Classical IR Metrics + RAGAS | Quantitative retrieval & generation assessment |

---

## 🏗️ Part 1 Architecture Pipeline

```text
┌─────────────────────────────────┐
│        2WikiMultiHopQA          │
│       Document Corpus           │
└────────────────┬────────────────┘
                 │
                 ▼
     [Document Ingestion & Cleaning]
                 │
                 ▼
     [Recursive Text Chunking]
    (500-800 tokens, 10-20% overlap)
                 │
                 ▼
   [Metadata Attachment & bge-m3 Embeddings]
                 │
                 ▼
            [Qdrant DB]
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
[Dense Retrieval]     [BM25 Retrieval]
 (Cosine Sim Top-K)   (Keyword Top-K)
      └──────────┬──────────┘
                 ▼
       [RRF Rank Fusion]
                 │
                 ▼
       [Candidate Pool]
                 │
                 ▼
   [Cross-Encoder Reranker]
   (BAAI/bge-reranker-v2-m3)
                 │
                 ▼
      [Top-N Best Evidence]
                 │
                 ▼
         [Groq LLM Engine]
                 │
                 ▼
   [Grounded Answer + Citations]
                 │
                 ▼
     [RAG Evaluation Pipeline]
  (IR Metrics: Recall, MRR, nDCG
   RAGAS: Faithfulness, Correctness)
```

---

## 🧪 Experimental Ablation Plan

The RAG baseline must be evaluated across 4 configurations against a **Curated Golden Evaluation Set (~1,000 multi-hop questions)**:

1. **Experiment 1 (BM25 Only)**: `Query → BM25 → Top-K → Groq LLM`
2. **Experiment 2 (Dense Only)**: `Query → BGE-M3 + Qdrant → Top-K → Groq LLM`
3. **Experiment 3 (Hybrid RAG)**: `Dense + BM25 → RRF Fusion → Top-K → Groq LLM`
4. **Experiment 4 (Final Baseline)**: `Dense + BM25 → RRF Fusion → BGE Reranker v2 M3 → Top-N → Groq LLM`

---

## 📊 Evaluation Metrics

### Retrieval Evaluation
- **Recall@K** ($K \in \{1, 3, 5, 10\}$)
- **Precision@K**
- **MRR** (Mean Reciprocal Rank)
- **nDCG@5** & **nDCG@10**

### Generation & Grounding Evaluation
- **Answer Correctness** (vs. Gold Answers)
- **Faithfulness** (Hallucination / Context Grounding)
- **Answer Relevancy**
- **Context Precision & Recall**

---

## 🔮 Part 2: Agentic Layer (Roadmap)
Once Part 1 baseline metrics are established, Part 2 will build on top:
- Query Planning & Sub-query Routing Agent
- Evidence Verification & Sufficiency Agent
- Real-time Hallucination Detector
- Failure Analysis & Query Recovery Agent
- Adaptive Feedback Loop & Self-Improvement Memory

---

## 📂 Project Documentation
- Complete project summary: [PROJECT_SUMMARY.md](file:///e:/MultimodalRAG_EDAI/PROJECT_SUMMARY.md)
- Agent context rule: [.agents/rules/project_specification.md](file:///e:/MultimodalRAG_EDAI/.agents/rules/project_specification.md)
