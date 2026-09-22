# Complete Summary — RAG Part of Your Project


Your overall project is:

**Self-Improving Multi-Agent RAG with Hallucination Detection**

You decided to divide it into **two major parts**:

PART 1 → RAG

PART 2 → Agentic AI

This summary covers **only Part 1 — RAG**, while keeping the design compatible with the Agentic AI layer that will be added later.


## 1. Objective of the RAG Part


The goal of Part 1 is to build a **strong, measurable hybrid RAG baseline**.

The RAG system should:

Ingest documents

Clean and process text

Split documents into chunks

Generate embeddings

Store embeddings in a vector database

Perform dense semantic retrieval

Perform sparse keyword retrieval using BM25

Combine retrieval results using RRF

Rerank retrieved documents using a cross-encoder

Give the best evidence to an LLM

Generate an answer grounded in the retrieved evidence

Return the answer along with source information

Be quantitatively evaluated

The Agentic AI part will later sit **on top of this RAG system** and add intelligent routing, verification, hallucination detection, recovery and self-improvement.


## 2. Final Technology Stack


Use this stack for the RAG part:

| Component | Technology |

| --- | --- |

| Programming Language | Python |

| RAG Framework | LangChain |

| LLM Provider | Groq |

| LLM | Specific Groq-supported model |

| Vector Database | Qdrant |

| Dense Embeddings | BAAI/bge-m3 |

| Sparse Retrieval | BM25 |

| Chunking | Recursive Character Text Splitter |

| Retrieval Fusion | RRF — Reciprocal Rank Fusion |

| Reranker | BAAI/bge-reranker-v2-m3 |

| Evaluation | RAGAS + classical IR metrics |

**Do NOT use in Part 1**

CrewAI

LangGraph Agents

Query Agent

Evidence Agent

Recovery Agent

Hallucination Agent

Self-Improvement Agent

Those belong primarily to **Part 2 — Agentic AI**.

You can use LangGraph later for the agentic orchestration layer, but it isn't necessary for the basic RAG pipeline.


## 3. Vector Database Decision


Initially you considered:

ChromaDB / Qdrant

The final recommendation is:

**Qdrant**

Reason: your architecture uses **dense + sparse/hybrid retrieval + RRF + reranking**, making Qdrant a better fit for the planned system.

So don't keep the requirement as:

ChromaDB/Qdrant

Use:

**Vector Database:****Qdrant**


## 4. Embedding Model


Don't write simply:

BGE

Specify the model.

Recommended:

**BAAI/bge-m3**

So:

Embedding Model

↓

BAAI/bge-m3

If you later decide the dataset is strictly English and want a smaller model, an English BGE model can also be considered, but **bge-m3 is the current recommended choice for your architecture**.


## 5. Chunking


Use:

**Recursive Character Text Splitting**

with configurable:

Chunk size

Chunk overlap

Don't hard-code the values without experimentation.

A reasonable starting point is:

Chunk size: ~500–800 tokens

Overlap: ~10–20%

Then evaluate different configurations.

Each chunk should preserve metadata.

Example:

{

"document_id": "...",

"chunk_id": "...",

"title": "...",

"source": "...",

"section": "...",

"page": "...",

"text": "..."

}

Metadata is important for source attribution and later Agentic AI/hallucination verification.


## 7. Do You Need a Separate Golden Dataset?


A separate generic golden dataset is **not strictly necessary** because 2WikiMultiHopQA already contains ground-truth answers and supporting evidence.

However, for your project, create a **curated Golden Evaluation Set** from 2WikiMultiHopQA.

Recommended starting point:

**1,000 questions**

Your golden evaluation set should contain:

question

gold_answer

gold_documents

gold_passages / supporting_facts

question_type

sample_id

Conceptually:

{

"question": "...",

"gold_answer": "...",

"gold_documents": ["doc1", "doc2"],

"gold_passages": ["...", "..."],

"question_type": "...",

"sample_id": "..."

}

This gives you a controlled benchmark for comparing your different RAG approaches.


## 8. Dataset Split


Don't tune your system using the final test set.

Use:

Training / Development

↓

Parameter and pipeline development

Validation

↓

Tune chunking, Top-K, thresholds, etc.

Test / Golden Evaluation Set

↓

Final evaluation

Initially, a manageable setup could be:

Development → 10,000–20,000 examples

Validation → 1,000–2,000

Golden Test → ~1,000

You don't need to process the entire dataset initially.


## 9. Document Ingestion Pipeline


The first part of the architecture is:

2WikiMultiHopQA

│

↓

Document Loader

│

↓

Text Extraction

│

↓

Text Cleaning

│

↓

Recursive Chunking

│

↓

Metadata Creation

│

↓

BGE-M3

Embeddings

│

↓

Qdrant

The important thing is that the **document corpus** and **question/answer evaluation set** are conceptually separate.


## 10. Dense Retrieval


Your first retrieval method is semantic retrieval.

User Question

↓

BGE-M3 Embedding

↓

Qdrant

↓

Cosine/Vector Similarity

↓

Top-K Documents

This retrieves documents based on semantic meaning.

Example:

Query:

"Who established the organization?"

Relevant text:

"The organization was founded by..."

Even if the exact word "established" doesn't appear, semantic retrieval can identify the relevant passage.


## 11. Sparse Retrieval — BM25


Your second retrieval method is:

**BM25**

Pipeline:

Query

↓

BM25

↓

Keyword matching

↓

Top-K documents

BM25 is particularly useful when the query contains:

Names

Dates

Exact terminology

Technical terms

Proper nouns

Specific phrases

This complements dense semantic retrieval.


## 12. Hybrid Retrieval


Your main RAG improvement is:

QUERY

│

┌────────┴────────┐

↓                 ↓

Dense Retrieval       BM25

BGE + Qdrant        Sparse Search

↓                 ↓

Top-K              Top-K

└────────┬────────┘

↓

RRF

↓

Candidate Pool

This is your **Hybrid RAG**.


## 13. RRF — Reciprocal Rank Fusion


After dense retrieval and BM25 retrieval, combine their rankings using:

**Reciprocal Rank Fusion (RRF)**

Conceptually:

Dense Results

+

BM25 Results

↓

RRF

↓

Unified Ranking

The purpose is to combine:

Semantic relevance from dense retrieval

Exact lexical relevance from BM25

This should be one of your main experimental improvements.


## 14. Cross-Encoder Reranking


After RRF, don't immediately send all documents to the LLM.

Use:

**BAAI/bge-reranker-v2-m3**

Pipeline:

Dense Retrieval

+

BM25

↓

RRF

↓

Candidate Pool

↓

Cross Encoder

↓

bge-reranker-v2-m3

↓

Re-ranked Evidence

↓

Top-N

The cross-encoder evaluates the query and candidate passage together and provides a more detailed relevance score.

This gives you:

**Hybrid Retrieval + Reranking**

which should be your final RAG configuration.


## 15. Final RAG Architecture


The complete RAG-only architecture is:

┌─────────────────────┐

│  2WikiMultiHopQA    │

│   DOCUMENT CORPUS   │

└──────────┬──────────┘

↓

Document Loader

↓

Text Cleaning

↓

Recursive Chunking

↓

Metadata

↓

BAAI/bge-m3

Embedding

↓

Qdrant

Vector DB

│

┌──────────┴──────────┐

↓                     ↓

Dense Retrieval            BM25

BGE + Qdrant          Sparse Retrieval

↓                     ↓

Top-K                 Top-K

└──────────┬──────────┘

↓

RRF Fusion

↓

Candidate Pool

↓

BAAI/bge-reranker-v2-m3

Cross Encoder

↓

Top-N

Best Evidence

↓

Groq LLM

↓

Final Answer

↓

Answer + Sources


## 16. RAG Evaluation


Your RAG evaluation should have **two major sections**:

RAG Evaluation

│

├───────────────┐

↓               ↓

Retrieval          Generation

Evaluation         Evaluation


## 17. Retrieval Evaluation


Use the supporting_facts / gold evidence from 2WikiMultiHopQA.

Measure:

**Recall@K**

Did the system retrieve the required evidence?

Evaluate:

Recall@1

Recall@3

Recall@5

Recall@10

**Precision@K**

How many retrieved documents are actually relevant?

For example:

Gold:

D12, D48

Retrieved:

D12, D31, D48, D72, D91

Then:

Precision@5 = 2/5

**MRR**

Mean Reciprocal Rank

Measures how high the first relevant document appears.

Useful for comparing:

BM25

vs

Dense

vs

Hybrid

vs

Hybrid + Reranker

**nDCG@K**

Measures the quality of the ranking, not just whether relevant documents were retrieved.

Use:

nDCG@5

nDCG@10

especially because your system has a reranker.


## 18. Generation Evaluation


After retrieval:

Retrieved Evidence

↓

Groq

↓

Generated Answer

Evaluate:

**Answer Correctness**

Compare generated answer against:

Gold Answer

**Faithfulness**

Check whether the generated answer is supported by the retrieved context.

**Answer Relevancy**

Check whether the answer actually answers the question.

**Context Precision**

Check whether retrieved context is relevant.

**Context Recall**

Check whether the retrieved context contains the necessary information.


## 19. Use RAGAS


For generation/RAG evaluation, use:

**RAGAS**

Recommended metrics:

Faithfulness

Answer Relevancy

Context Precision

Context Recall

Answer Correctness

But don't rely only on RAGAS.

Use:

Classical IR metrics

+

RAGAS metrics

So your evaluation becomes:

Retrieval:

Recall@K

Precision@K

MRR

nDCG

Generation:

Answer Correctness

Faithfulness

Answer Relevancy

Context Precision

Context Recall


## 20. Most Important Experimental Design


You should **not only build the final Hybrid +****Reranker****system**.

Perform an ablation/comparison study.


### Experiment 1 — BM25


Query

↓

BM25

↓

Top-K

↓

Groq


### Experiment 2 — Dense RAG


Query

↓

BGE-M3

↓

Qdrant

↓

Top-K

↓

Groq


### Experiment 3 — Hybrid RAG


BGE + Qdrant

+

BM25

↓

RRF

↓

Top-K

↓

Groq


### Experiment 4 — Final RAG


BGE + Qdrant

+

BM25

↓

RRF

↓

Candidate Pool

↓

BGE Reranker

↓

Top-N

↓

Groq


## 21. Comparison Table for Your Research


Your final report should contain something like:

**Retrieval**

| Method | Recall@5 | Recall@10 | MRR | nDCG@10 |

| --- | --- | --- | --- | --- |

| BM25 | Actual result | Actual result | Actual result | Actual result |

| BGE Dense | Actual result | Actual result | Actual result | Actual result |

| BGE + BM25 | Actual result | Actual result | Actual result | Actual result |

| BGE + BM25 + RRF | Actual result | Actual result | Actual result | Actual result |

| Hybrid + Reranker | Actual result | Actual result | Actual result | Actual result |

**Generation**

| Method | Answer Correctness | Faithfulness | Relevancy |

| --- | --- | --- | --- |

| Dense RAG | Actual result | Actual result | Actual result |

| Hybrid RAG | Actual result | Actual result | Actual result |

| Hybrid + Reranker | Actual result | Actual result | Actual result |

Do **not** invent these numbers; populate them only after running the experiments.


## 22. Why this is important for Part 2


Your RAG part should become the **baseline infrastructure** for the Agentic AI part.

The separation should look like:

PART 1

STATIC RAG

│

↓

┌───────────────────────┐

│ Dense + BM25 + RRF    │

│ + Cross Encoder       │

│ + Groq                │

└───────────┬───────────┘

│

↓

PART 2

AGENTIC AI LAYER

│

┌─────────┼──────────┐

↓         ↓          ↓

Query     Evidence    Answer

Agent      Agent       Agent

│         │          │

└─────────┼──────────┘

↓

Hallucination Detector

↓

Failure Analysis

↓

Recovery Agent

↓

Retry

↓

Self-Improvement

So **Part 1 should not try to solve everything**.

Its job is to provide a **high-quality, measurable RAG foundation**.


## 23. What NOT to add to RAG Part


Keep these for Part 2:

❌ Query Agent

❌ Retrieval Strategy Agent

❌ Evidence Agent

❌ Recovery Agent

❌ Hallucination Agent

❌ Failure Classifier

❌ Self-improvement memory

❌ Adaptive strategy selection

❌ Autonomous retries

Part 1 should remain:

Documents

↓

Chunking

↓

Embedding

↓

Qdrant

↓

Dense Retrieval

+

BM25

↓

RRF

↓

Reranking

↓

Groq

↓

Answer + Sources


## 24. Final RAG Part Specification


You can directly use the following in your project proposal:

**RAG Component**

**Dataset:**2WikiMultiHopQA

**Golden Evaluation Set:**A curated subset of approximately 1,000 questions containing ground-truth answers and supporting evidence.

**Document Processing:**Document loading → text cleaning → recursive chunking → metadata extraction.

**Embedding:**BAAI/bge-m3

**Vector Database:**Qdrant

**Sparse Retrieval:**BM25

**Hybrid Retrieval:**Dense BGE retrieval + BM25 → Reciprocal Rank Fusion (RRF)

**Reranking:**BAAI/bge-reranker-v2-m3

**LLM:**Groq-hosted LLM

**Framework:**Python + LangChain

**Output:**Grounded answer + source/document references.

**Retrieval Evaluation:**

Recall@K

Precision@K

MRR

nDCG@K

**Generation/RAG Evaluation:**

Answer Correctness

Faithfulness

Answer Relevancy

Context Precision

Context Recall

**Experiments:**

BM25

↓

Dense BGE

↓

Dense + BM25

↓

Dense + BM25 + RRF

↓

Dense + BM25 + RRF + Reranker

The **last configuration becomes your RAG baseline** for Part 2.

**Final architecture to implement**

┌──────────────────┐

│ 2WikiMultiHopQA  │

│ Document Corpus  │

└────────┬─────────┘

↓

Document Processing

↓

Recursive Chunking

↓

Metadata

↓

BAAI/bge-m3

Embedding

↓

Qdrant

│

┌─────────────┴─────────────┐

↓                           ↓

Dense Retrieval                  BM25

↓                           ↓

Top-K                       Top-K

└─────────────┬─────────────┘

↓

RRF Fusion

↓

Candidate Pool

↓

BAAI/bge-reranker-v2-m3

Cross Encoder

↓

Top-N

Best Evidence

↓

Groq LLM

↓

Answer + Citations

│

↓

┌─────────────────────────┐

│    RAG EVALUATION       │

│                         │

│ Recall@K                │

│ Precision@K             │

│ MRR                     │

│ nDCG@K                  │

│ Faithfulness            │

│ Answer Correctness      │

│ Answer Relevancy        │

│ Context Precision       │

│ Context Recall          │

└─────────────────────────┘

**This is the RAG part I recommend you actually build first.** Once this works and you have baseline metrics, the Agentic AI portion can be added on top without changing the core retrieval infrastructure.