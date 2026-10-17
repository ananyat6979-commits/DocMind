"""
End-to-end demo: ingest a PDF/DOCX and ask questions about it.
Usage: python scripts/demo.py --doc data/your_document.pdf
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")

from docmind.ingestion.pipeline import ingest
from docmind.agent.react import ReActAgent


def main():
    parser = argparse.ArgumentParser(description="DocMind demo")
    parser.add_argument("--doc", required=True, help="Path to document or directory")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild indexes from scratch")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("DocMind - Intelligent Document Intelligence")
    print(f"{'='*60}\n")

    # Ingest
    print(f"Ingesting: {args.doc}")
    report = ingest(args.doc, rebuild_index=args.rebuild)
    print(f"Ingested {report.chunks_added} new chunks ({report.n_cached} files already cached)\n")

    # Interactive Q&A
    agent = ReActAgent()
    file_ids = report.file_ids or None
    print("Ask questions about your document. Type 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\nThinking...\n")
        response = agent.run(question, file_ids=file_ids)

        print(f"Answer: {response.answer}\n")
        if response.needs_review:
            print(f"[flagged for human review: {', '.join(response.review_reasons)}]\n")
        print(f"Sources ({len(response.sources)}):")
        for s in response.sources:
            meta = s.chunk.metadata
            loc = f"p.{meta['page']}" if "page" in meta else meta.get("section", "")
            print(f"  • {meta.get('filename', s.chunk.source)} {loc} (score: {s.score:.3f})")
        print(f"\n[{response.latency_ms:.0f}ms | {len(response.reasoning_trace)} reasoning steps]\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
