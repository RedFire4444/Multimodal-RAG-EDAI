import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

import logging

# Suppress verbose third-party logging (httpx HTTP request logs, etc.)
logging.basicConfig(level=logging.WARNING)
for _logger_name in ("httpx", "httpcore", "groq", "flashrank", "sentence_transformers", "qdrant_client", "urllib3"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

# Safe UTF-8 configuration for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables
load_dotenv()

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.service import RAGService
from src.evaluation.dataset import GoldenDatasetLoader
from src.evaluation.evaluator import BenchmarkEvaluator

console = Console(force_terminal=True, legacy_windows=False)


def run_ingest(path: str = "data", limit: int = None):
    console.print(Panel(f"[bold cyan]Starting Ingestion Pipeline[/bold cyan]\nTarget: {path}\nLimit: {limit}"))
    service = RAGService()
    count = service.ingest_and_index(file_path=path, limit=limit)
    console.print(f"[bold green]Successfully indexed {count} chunks into Qdrant & BM25![/bold green]")


def display_rag_response(response, show_evidence: bool = True):
    console.print("\n[bold green]=== Grounded Answer ===[/bold green]")
    console.print(Markdown(response.answer))

    if show_evidence and response.evidence_chunks:
        console.print(f"\n[bold cyan]=== Retrieved Evidence ({len(response.evidence_chunks)} passages) ===[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", width=6)
        table.add_column("Source", width=12)
        table.add_column("Document", width=22)
        table.add_column("Score", width=8)
        table.add_column("Text Preview", width=55)

        for chunk in response.evidence_chunks:
            table.add_row(
                str(chunk.rank),
                chunk.source_method,
                chunk.title or "N/A",
                f"{chunk.score:.4f}",
                (chunk.text[:120] + "...") if len(chunk.text) > 120 else chunk.text,
            )
        console.print(table)

    retrieval_sec = response.metadata.get("retrieval_latency_sec", 0)
    gen_sec = response.metadata.get("generation_latency_sec", 0)
    tot_sec = response.metadata.get("total_latency_sec", 0)
    console.print(f"[dim]Latency: Retrieval={retrieval_sec}s | Generation={gen_sec}s | Total={tot_sec}s[/dim]\n")


def execute_streaming_query(service: RAGService, query_text: str, top_k: int = 5, show_evidence: bool = True):
    """Executes query with real-time token streaming and evidence display."""
    evidence_chunks = []
    final_response = None

    with console.status("[bold cyan]Retrieving & reranking evidence...[/bold cyan]", spinner="dots"):
        stream = service.query_stream(query=query_text, top_k=top_k)
        event_type, evidence_chunks, ret_latency = next(stream)

    console.print("\n[bold green]=== Grounded Answer ===[/bold green]")
    for event_type, data, extra in stream:
        if event_type == "token":
            console.print(data, end="", highlight=False)
        elif event_type == "done":
            final_response = extra

    console.print("\n")

    if final_response:
        if show_evidence and final_response.evidence_chunks:
            console.print(f"[bold cyan]=== Retrieved Evidence ({len(final_response.evidence_chunks)} passages) ===[/bold cyan]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Rank", width=6)
            table.add_column("Source", width=12)
            table.add_column("Document", width=22)
            table.add_column("Score", width=8)
            table.add_column("Text Preview", width=55)

            for chunk in final_response.evidence_chunks:
                table.add_row(
                    str(chunk.rank),
                    chunk.source_method,
                    chunk.title or "N/A",
                    f"{chunk.score:.4f}",
                    (chunk.text[:120] + "...") if len(chunk.text) > 120 else chunk.text,
                )
            console.print(table)

        retrieval_sec = final_response.metadata.get("retrieval_latency_sec", 0)
        gen_sec = final_response.metadata.get("generation_latency_sec", 0)
        tot_sec = final_response.metadata.get("total_latency_sec", 0)
        console.print(f"[dim]Latency: Retrieval={retrieval_sec}s | Generation={gen_sec}s | Total={tot_sec}s[/dim]\n")


def run_query(query_text: str, top_k: int = 5):
    console.print(Panel(f"[bold yellow]Executing Hybrid RAG Query[/bold yellow]\nQuery: {query_text}\nTop-K Evidence: {top_k}"))
    service = RAGService()
    execute_streaming_query(service, query_text=query_text, top_k=top_k)


def run_interactive_chat(top_k: int = 5):
    console.print(Panel.fit(
        "[bold cyan]Hybrid RAG Interactive Q&A Assistant[/bold cyan]\n"
        "[dim]Ask questions against your knowledge base in data/. Models stay loaded in memory for fast streaming answers.[/dim]\n"
        "[bold yellow]Type 'exit', 'quit', or 'q' to stop.[/bold yellow]",
        border_style="cyan"
    ))

    with console.status("[bold cyan]Loading models and warming up into memory...[/bold cyan]", spinner="dots"):
        service = RAGService()
        service.warmup()
    
    bm25_status = f"{len(service.bm25_retriever.chunks)} chunks in cache" if service.bm25_retriever.chunks else "Empty (run ingest first)"
    console.print(f"[dim]Dense Vectors: Ready | BM25 Index: {bm25_status} | LLM: Ready[/dim]")
    console.print("[bold green]System ready! Type your question below:[/bold green]\n")

    while True:
        try:
            query = console.input("[bold cyan]Question:[/bold cyan] ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break

            execute_streaming_query(service, query_text=query, top_k=top_k)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error answering query:[/bold red] {e}\n")



def run_benchmark(golden_dataset_path: str, limit: int = 100):
    console.print(Panel(f"[bold magenta]Starting Benchmark Evaluation[/bold magenta]\nDataset: {golden_dataset_path}\nSample Limit: {limit}"))
    service = RAGService()
    samples = GoldenDatasetLoader.load_golden_set(golden_dataset_path, limit=limit)
    evaluator = BenchmarkEvaluator(pipeline=service.pipeline)

    res = evaluator.evaluate_retrieval(samples=samples, top_k=10)

    table = Table(title="Hybrid RAG Benchmark Results: Retrieval", header_style="bold green")
    table.add_column("Metric", style="bold")
    table.add_column("Score", style="cyan")

    table.add_row("Recall@1", f"{res['Recall@1']:.4f}")
    table.add_row("Recall@3", f"{res['Recall@3']:.4f}")
    table.add_row("Recall@5", f"{res['Recall@5']:.4f}")
    table.add_row("Recall@10", f"{res['Recall@10']:.4f}")
    table.add_row("MRR", f"{res['MRR']:.4f}")
    table.add_row("nDCG@5", f"{res['nDCG@5']:.4f}")
    table.add_row("nDCG@10", f"{res['nDCG@10']:.4f}")

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Multi-Agent RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Interactive chat command (default when no sub-command given)
    chat_parser = subparsers.add_parser("chat", help="Start interactive Q&A session")
    chat_parser.add_argument("--top_k", type=int, default=5, help="Number of evidence chunks (default: 5)")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from data/ folder or file")
    ingest_parser.add_argument("--path", type=str, default="data", help="Path to data directory or file (default: data)")
    ingest_parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to ingest")

    # Single Query command
    query_parser = subparsers.add_parser("query", help="Query the Hybrid RAG pipeline directly")
    query_parser.add_argument("--text", type=str, required=True, help="Question text")
    query_parser.add_argument("--top_k", type=int, default=5, help="Number of evidence chunks (default: 5)")

    # Benchmark command
    eval_parser = subparsers.add_parser("benchmark", help="Run benchmark evaluation on Golden dataset")
    eval_parser.add_argument("--golden_set", type=str, required=True, help="Path to golden evaluation json")
    eval_parser.add_argument("--limit", type=int, default=100, help="Number of questions to evaluate")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(path=args.path, limit=args.limit)
    elif args.command == "query":
        run_query(query_text=args.text, top_k=args.top_k)
    elif args.command == "benchmark":
        run_benchmark(golden_dataset_path=args.golden_set, limit=args.limit)
    elif args.command == "chat":
        run_interactive_chat(top_k=args.top_k)
    else:
        run_interactive_chat()


if __name__ == "__main__":
    main()
