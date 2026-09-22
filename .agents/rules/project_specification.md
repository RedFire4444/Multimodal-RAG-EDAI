# Project Specification: Self-Improving Multi-Agent RAG with Hallucination Detection

## Project Overview
The project is divided into two major phases:
- **Part 1: RAG Baseline** (Dense + Sparse BM25 + RRF Fusion + Cross-Encoder Reranking + Groq LLM + Benchmark Evaluation).
- **Part 2: Agentic AI Layer** (Intelligent routing, multi-agent query/evidence verification, hallucination detection, recovery, and self-improvement).

> **Important**: All immediate work is focused strictly on **Part 1 (RAG Baseline)**. Do not introduce autonomous agents (CrewAI, LangGraph Agent workflows, multi-agent retries) until the RAG baseline and ablation evaluations are complete and benchmarked.

---

## Part 1 Technology Stack & Decisions
- **Language**: Python 3.10+
- **RAG Framework**: LangChain
- **LLM Provider**: Groq (Fast inference)
- **Vector Database**: Qdrant (Supports dense vectors, payload filtering, hybrid retrieval)
- **Dense Embeddings**: `BAAI/bge-m3`
- **Sparse Retrieval**: BM25
- **Retrieval Fusion**: Reciprocal Rank Fusion (RRF)
- **Reranker**: Cross-Encoder `BAAI/bge-reranker-v2-m3`
- **Dataset**: 2WikiMultiHopQA (Multi-hop question answering with supporting facts)
- **Golden Evaluation Set**: Curated subset of ~1,000 multi-hop questions with ground-truth answers and supporting passages/documents.
- **Evaluation Framework**: RAGAS + Classical Information Retrieval (IR) metrics.

---

## Pipeline Architecture
1. **Document Ingestion**:
   - 2WikiMultiHopQA Corpus → Text Cleaning → Recursive Character Text Splitting (chunk size ~500–800 tokens, 10–20% overlap).
   - Rich metadata preservation (`document_id`, `chunk_id`, `title`, `source`, `section`, `page`, `text`).
   - Embedding generation via `BAAI/bge-m3` → Stored in **Qdrant**.
2. **Hybrid Retrieval**:
   - Query → Dense Retrieval (`bge-m3` + Qdrant cosine similarity Top-K)
   - Query → Sparse Retrieval (BM25 keyword matching Top-K)
   - Combined via **Reciprocal Rank Fusion (RRF)** into a candidate pool.
3. **Cross-Encoder Reranking**:
   - Candidate pool scored by `BAAI/bge-reranker-v2-m3` → Select Top-N evidence chunks.
4. **Generation**:
   - Groq-hosted LLM receives top evidence passages and query → Generates grounded response with explicit source citations.

---

## Experimental Ablation Setup
The project must systematically evaluate and record metrics for 4 configurations:
1. **Experiment 1 (BM25 Only)**: BM25 → Top-K → Groq LLM
2. **Experiment 2 (Dense Only)**: BGE-M3 + Qdrant → Top-K → Groq LLM
3. **Experiment 3 (Hybrid RAG)**: Dense (BGE) + BM25 → RRF Fusion → Top-K → Groq LLM
4. **Experiment 4 (Final RAG Baseline)**: Dense + BM25 → RRF Fusion → BGE Reranker v2 M3 → Top-N → Groq LLM

---

## Evaluation Metrics
- **Retrieval Metrics**:
  - `Recall@1`, `Recall@3`, `Recall@5`, `Recall@10`
  - `Precision@K`
  - `MRR` (Mean Reciprocal Rank)
  - `nDCG@5`, `nDCG@10`
- **Generation Metrics (RAGAS & Classical)**:
  - `Answer Correctness`
  - `Faithfulness` (Groundedness / Hallucination check)
  - `Answer Relevancy`
  - `Context Precision` & `Context Recall`
