FROM python:3.11-slim as builder

WORKDIR /code

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"


COPY demo_target_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 
    # && \
    # pip install --no-cache-dir sentence-transformers einops --extra-index-url https://download.pytorch.org/whl/cpu

# ENV HF_HOME=/root/.cache/huggingface
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"
#Commented these lines, bec sentence transformers are no longer required

ENV FASTEMBED_CACHE_PATH  = /models/fastembed_cache

RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; TextEmbedding('nomic-ai/nomic-embed-text-v1.5', cache_dir='/models/fastembed_cache'); SparseTextEmbedding('Qdrant/bm25', cache_dir='/models/fastembed_cache')"


FROM python:3.11-slim as runner

WORKDIR /code


COPY --from=builder /opt/venv /opt/venv

# ENV HF_HOME=/root/.cache/huggingface
ENV FASTEMBED_CACHE_PATH  = /models/fastembed_cache

# COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY --from=builder /models/fastembed_cache /models/fastembed_cache

COPY demo_target_app/ ./demo_target_app/

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/code

EXPOSE 8000

CMD ["uvicorn", "demo_target_app.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]