# 🚀 Multimodal Hybrid RAG with Hallucination Detection & FlashRank

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/orchestration-LangChain-green.svg)](https://www.langchain.com/)
[![Qdrant](https://img.shields.io/badge/vectorstore-Qdrant-red.svg)](https://qdrant.tech/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)
[![FlashRank](https://img.shields.io/badge/reranker-FlashRank-yellow.svg)](https://github.com/PrithivirajDamodaran/FlashRank)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade, high-performance **Hybrid Retrieval-Augmented Generation (RAG)** pipeline designed for multi-hop reasoning and strict factual grounding. Combines dense semantic representations and sparse lexical search via **Reciprocal Rank Fusion (RRF)**, refined by ultra-fast **FlashRank ONNX Reranking**, and powered by **Groq LPU LLM inference**.

---

## 📑 Table of Contents
- [Architecture & Pipeline](#-architecture--pipeline)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Environment Configuration](#-environment-configuration)
- [How to Run the Project](#-how-to-run-the-project)
  - [1. Ingest & Index Documents](#1-ingest--index-documents)
  - [2. Interactive Terminal Chat (Streaming)](#2-interactive-terminal-chat-streaming)
  - [3. Single Direct Query](#3-single-direct-query)
  - [4. Benchmark Evaluation](#4-benchmark-evaluation)
- [Evaluation Metrics](#-evaluation-metrics)
- [Roadmap](#-roadmap)

---

## 🏗️ Architecture & Pipeline

```
                     ┌───────────────────────────────┐
                     │   Raw Documents (PDF / TXT)   │
                     └───────────────┬───────────────┘
                                     │
                     ┌───────────────▼───────────────┐
                     │ Ingestion & Recursive Chunker │
                     │   (500-1000 tokens, overlap)  │
                     └───────┬───────────────┬───────┘
                             │               │
                             ▼               ▼
                   ┌──────────────────┐  ┌──────────────────┐
                   │ Sentence Embeds  │  │   BM25 Corpus    │
                   │   (Qdrant DB)    │  │   Token Index    │
                   └─────────┬────────┘  └─────────┬────────┘
                             │                     │
                     Dense Top-20              Sparse Top-20
                             │                     │
                             └─────────┬───────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │   Reciprocal Rank Fusion  │
                         │      (RRF Top 15-20)      │
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │  FlashRank ONNX Reranker  │
                         │    (ms-marco-TinyBERT)    │
                         └─────────────┬─────────────┘
                                       ▼
                            Top-N Evidence Passages
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │      Groq LPU Engine      │
                         │ (llama-3.3-70b-versatile) │
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │   Grounded Answer with    │
                         │   Citations & Evidence    │
                         └───────────────────────────┘
```

---

## ✨ Key Features

- **Hybrid Dual-Stream Retrieval**:
  - **Dense Search**: Cosine similarity via `sentence-transformers` and cloud-hosted **Qdrant Vector Database**.
  - **Sparse Search**: Keyword, entity, and lexical matching powered by **BM25** with disk-persisted caching.
- **Reciprocal Rank Fusion (RRF)**: Merges heterogeneous scoring spaces into a balanced candidate pool without calibration artifacts.
- **Ultra-Fast FlashRank Reranking**: Light-weight, CPU-native ONNX cross-encoder (`ms-marco-TinyBERT-L-2-v2`) delivering sub-millisecond reranking latency without heavy PyTorch GPU overhead.
- **Sub-Second LLM Generation**: Grounded generation with real-time token streaming powered by **Groq** (`llama-3.3-70b-versatile` / `openai/gpt-oss-120b`).
- **Comprehensive Evaluation Benchmark**: Complete quantitative assessment suite supporting Classical Information Retrieval (IR) metrics (Recall@K, MRR, nDCG) and RAGAS metrics.

---

## 📂 Project Structure

```text
Multimodal-RAG-EDAI/
├── app/
│   └── main.py                     # Rich interactive CLI entry point
├── data/
│   ├── sample_kb.txt               # Sample knowledge base document
│   └── *.pdf                       # Ingestion target documents
├── src/
│   ├── embeddings/
│   │   └── bge_embeddings.py       # Dense embedding wrapper
│   ├── evaluation/
│   │   ├── dataset.py              # Golden dataset loader (2WikiMultiHopQA)
│   │   ├── evaluator.py            # Benchmark evaluation runner
│   │   ├── generation_metrics.py   # Exact Match, F1, ROUGE-L, BLEU
│   │   ├── ragas_evaluator.py      # RAGAS faithfulness & answer relevance
│   │   └── retrieval_metrics.py    # Recall@K, MRR, nDCG@K
│   ├── generation/
│   │   ├── answer_generator.py     # Grounded LLM response generator
│   │   ├── groq_client.py          # Groq API client with failover
│   │   └── prompts.py              # Strict citation prompts
│   ├── ingestion/
│   │   ├── chunker.py              # Recursive text splitter
│   │   ├── cleaner.py              # Text cleaning & normalization
│   │   ├── loaders.py              # Multi-format document loader (PDF, TXT)
│   │   ├── metadata.py             # Metadata extraction & schema builder
│   │   └── pipeline.py             # Ingestion orchestration pipeline
│   ├── rag/
│   │   ├── pipeline.py             # End-to-end Hybrid RAG pipeline
│   │   ├── schemas.py              # Pydantic data models
│   │   └── service.py              # High-level RAG service layer
│   ├── retrieval/
│   │   ├── bm25_retriever.py       # BM25 sparse keyword retriever
│   │   ├── dense_retriever.py      # Dense vector retriever (Qdrant)
│   │   ├── hybrid_retriever.py     # Parallel hybrid retriever (Dense + BM25)
│   │   ├── reranker.py             # FlashRank ONNX Reranker
│   │   └── rrf.py                  # Reciprocal Rank Fusion algorithm
│   └── vectorstore/
│       └── qdrant_store.py         # Qdrant client connection & payload indexer
├── main.py                         # Root execution proxy
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
└── README.md                       # Project documentation
```

---

## ⚙️ Prerequisites & Installation

### 1. Clone the repository
```bash
git clone https://github.com/RedFire4444/Multimodal-RAG-EDAI.git
cd Multimodal-RAG-EDAI
```

### 2. Set up Python virtual environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

Create a `.env` file in the root directory by copying `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your API credentials and parameters:

```env
# Groq API Configuration
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Qdrant Vector Database
QDRANT_URL=https://your-cluster-id.region.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=MultimodalRAG
QDRANT_CLOUD_INFERENCE=false

# Embedding & Reranker Models
EMBEDDING_MODEL_NAME=sentence-transformers/all-minilm-l6-v2
EMBEDDING_DIM=384
RERANKER_MODEL_NAME=ms-marco-TinyBERT-L-2-v2

# Chunking Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval Defaults
TOP_K_DENSE=20
TOP_K_SPARSE=20
TOP_K_FUSED=15
TOP_N_RERANK=5
RRF_K=60
```

---

## 💻 How to Run the Project

### 1. Ingest & Index Documents
Parse and index all PDF and text documents in the `data/` folder into Qdrant vector database and BM25 search index:

```bash
# Ingest entire data/ folder
python main.py ingest --path data

# Ingest a specific document with a chunk limit
python main.py ingest --path data/sample_kb.txt --limit 50
```

---

### 2. Interactive Terminal Chat (Streaming)
Starts an interactive conversational session with real-time response streaming and evidence inspection tables. Models are pre-warmed in memory:

```bash
python main.py chat
```
*(Or simply run `python main.py`)*

Optional arguments:
- `--top_k <N>`: Adjust the number of reranked evidence passages retrieved (default: `5`).
```bash
python main.py chat --top_k 5
```

---

### 3. Single Direct Query
Execute a standalone query from the command line:

```bash
python main.py query --text "What are the key architectural advantages of Reciprocal Rank Fusion?"
```

---

### 4. Benchmark Evaluation
Evaluate retrieval and generation metrics across a Golden QA dataset:

```bash
python main.py benchmark --golden_set data/golden_qa.json --limit 100
```

---

## 📊 Evaluation Metrics

The evaluation suite supports:
- **Retrieval Performance**:
  - `Recall@K` ($K \in \{1, 3, 5, 10\}$)
  - `MRR` (Mean Reciprocal Rank)
  - `nDCG@5` and `nDCG@10`
- **Answer Quality & Grounding**:
  - `Exact Match (EM)` & `Token F1`
  - `ROUGE-L` & `BLEU`
  - `RAGAS Faithfulness` (Hallucination detection)
  - `RAGAS Answer Relevancy`

---

## 🗺️ Roadmap & Phase 2

- [x] **Part 1**: Hybrid RAG Baseline (Dense + BM25 + RRF + FlashRank ONNX + Groq)
- [x] **Part 1**: Ingestion Pipeline with Multi-Format Support (PDF, TXT)
- [x] **Part 1**: Live Token Streaming Interactive CLI Assistant
- [ ] **Part 2**: Multi-Hop Sub-Query Decomposition Agent
- [ ] **Part 2**: Real-time Hallucination Detection & Verification Agent
- [ ] **Part 2**: Adaptive Feedback Loop & Self-Improvement Query Reformulation
