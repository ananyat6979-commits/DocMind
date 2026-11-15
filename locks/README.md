# Reproducible environments

Production CPU uses `cpu-linux-py311.txt`. Windows CPU uses `cpu-windows-py311.txt`. Free notebooks use the Python 3.13 CPU or CUDA 13.0 lock as appropriate. Install with `pip install --require-hashes -r <lock> --extra-index-url https://download.pytorch.org/whl/cpu` for CPU, or replace the final index with `https://download.pytorch.org/whl/cu130` for CUDA.

Security maintenance pins Torch 2.13.0 and setuptools 84.2.1. Earlier published profiles record Torch 2.11.0 and retain their original provenance. A hash lock is not a compatibility result; consult release evidence for measured versions. The earlier FP16 T4 pilot does not validate the updated CUDA stack.

Regenerate with uv 0.13.8. For each CPU target, run:

```text
uv pip compile requirements-lock-cpu.in --python-version 3.11 --python-platform linux --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --generate-hashes --no-annotate --no-header --output-file locks/cpu-linux-py311.txt
```

Repeat using Python 3.13/Linux and Python 3.11/Windows with corresponding output filenames. For CUDA, use `requirements-lock-cuda.in`, Python 3.13/Linux, the `cu130` index and `locks/cuda-linux-py313.txt`. Inspect every package change and run compatibility checks and fresh affected baselines. Keep private advisory reports outside Git.


`postgres.txt` is an optional hash-locked Psycopg 3.3.9/binary overlay, installed after a locked CPU or CUDA environment. Its typing and timezone dependencies are already pinned in the base locks. The Docker image includes this overlay so one image supports either storage profile. Install it with `pip install --require-hashes -r locks/postgres.txt`. Database schema migrations use separate administrator credentials; merely installing the driver does not enable tenant mode.

The isolated parser has its own small `parser-linux-py311.txt` hash lock, resolved from `requirements-lock-parser.in`. Its image base is pinned by digest; this does not imply that a parser or dependency has no vulnerabilities.
