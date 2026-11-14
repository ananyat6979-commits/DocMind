# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    DOCMIND_HOME=/data \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only torch first. The default wheel drags in ~2 GB of CUDA libraries that are useless here.
COPY locks/cpu-linux-py311.txt /app/locks/cpu-linux-py311.txt
RUN pip install --require-hashes -r /app/locks/cpu-linux-py311.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .


# Bake the retrieval and attribution models INTO the image. Nothing is downloaded at start-up,
# it works offline, and the /data volume can never hide them (HF_HOME is outside /data).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5', revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', revision='233902d25c440f23af6f7d6e94d2946bac0bee0a')"
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; m='typeform/distilbert-base-uncased-mnli'; r='cfa538a0fddbbd978fefe8966c1aeff7ad409c90'; AutoTokenizer.from_pretrained(m, revision=r); AutoModelForSequenceClassification.from_pretrained(m, revision=r)"
ENV HF_HUB_OFFLINE=1

COPY docmind ./docmind
COPY app.py ./
COPY scripts ./scripts
COPY .streamlit ./.streamlit

# Run as a non-root user. /data (database, uploads, reports) is the only writable state.
RUN useradd --create-home --uid 10001 docmind \
 && mkdir -p /data \
 && chown -R docmind:docmind /data /app /opt/hf
USER docmind
EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read() == b'ok' else 1)"

# Configure proxy origins explicitly; access authentication and browser protections remain enabled.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.enableCORS=true", "--server.enableXsrfProtection=true"]
