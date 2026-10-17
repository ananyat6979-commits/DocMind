# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    DOCMIND_HOME=/data \
    HF_HOME=/opt/hf \
    NLTK_DATA=/opt/nltk_data

WORKDIR /app

# CPU-only torch first. The default wheel drags in ~2 GB of CUDA libraries that are useless here.
RUN pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the two models and the sentence tokenizer INTO the image. Nothing is downloaded at start-up,
# it works offline, and the /data volume can never hide them (HF_HOME is outside /data).
RUN python -c "import nltk; nltk.download('punkt', download_dir='/opt/nltk_data'); nltk.download('punkt_tab', download_dir='/opt/nltk_data')" \
 && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
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
VOLUME /data
EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read() == b'ok' else 1)"

# CORS/XSRF protection is off because the app is reached through a forwarding proxy (Codespaces, tunnels),
# where Streamlit's origin checks reject file uploads with a 403. Protect access with DOCMIND_APP_PASSWORD.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
