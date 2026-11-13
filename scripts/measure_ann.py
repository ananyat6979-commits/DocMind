"""Bounded synthetic FAISS ANN tradeoff; excludes embedding and service overhead."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DOCMIND_TEST_MODE", "1")
import faiss
import numpy as np
from measure_local import rss


def main():
    faiss.omp_set_num_threads(4)
    rng = np.random.default_rng(71)
    matrix = rng.standard_normal((50000, 384), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    queries = rng.standard_normal((100, 384), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    result = {"scope": "Synthetic normalized Gaussian vectors, not semantic retrieval or API load", "python": platform.python_version(),
              "faiss": faiss.__version__, "vectors": 50000, "dimensions": 384, "queries": 100,
              "seed": 71, "threads": 4, "top_k": 10, "vector_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(), "indexes": {}}
    truth = None
    for kind in ("flat", "hnsw"):
        index = faiss.IndexFlatIP(384) if kind == "flat" else faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)
        if kind == "hnsw":
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 64
        start = time.perf_counter()
        index.add(matrix)
        build = time.perf_counter() - start
        index.search(queries[:1], 10)
        timings = []
        predicted = []
        for query in queries:
            start = time.perf_counter()
            _, ids = index.search(query[None, :], 10)
            timings.append((time.perf_counter() - start) * 1000)
            predicted.append(ids[0].copy())
        if kind == "flat":
            truth = predicted
        recall = [len(set(a) & set(b)) / 10 for a, b in zip(predicted, truth)]
        result["indexes"][kind] = {"build_seconds": build, "serialized_bytes": len(faiss.serialize_index(index)),
            "p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95)),
            "recall10_against_exact": float(np.mean(recall)), "process_memory": rss()}
        print(kind, result["indexes"][kind], flush=True)
        del index
    result["limitations"] = ["Uniform vectors do not represent a real document distribution", "HNSW parallel construction may vary by hardware", "Memory is process cumulative peak, not isolated index RSS", "No confidence interval for single build or timing run"]
    (ROOT / "reports/ann_scale.json").write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
