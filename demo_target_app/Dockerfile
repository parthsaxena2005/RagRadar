FROM python:3.11-slim as builder

WORKDIR /code

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"


COPY demo_target_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sentence-transformers einops --extra-index-url https://download.pytorch.org/whl/cpu

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"


FROM python:3.11-slim as runner

WORKDIR /code


COPY --from=builder /opt/venv /opt/venv

COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

COPY demo_target_app/app ./app

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/code

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]