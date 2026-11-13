"""Download configured pinned models explicitly before starting an offline service."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from docmind.config import CONFIG, TEST_MODE
    if TEST_MODE:
        raise SystemExit("Turn off test mode before downloading live models")
    from docmind.embeddings import get_embedding_model
    from docmind.retrieval.reranker import _get_model
    from docmind.attribution import NLIJudge
    get_embedding_model()
    if CONFIG.retriever.use_reranker:
        _get_model()
    NLIJudge(model=os.getenv("DOCMIND_NLI_MODEL", "typeform/distilbert-base-uncased-mnli"),
             revision=os.getenv("DOCMIND_NLI_REVISION") or None, local_files_only=False)
    print("Configured retrieval and attribution models are cached. No provider request was sent.")


if __name__ == "__main__":
    main()
