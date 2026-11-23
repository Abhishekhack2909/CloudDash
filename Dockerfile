FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONTDONTWRITEBYTECODE=1 \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ingest KB at build time to precompute embeddings & BM25 indices
RUN python knowledge_base/ingest.py || true

EXPOSE 7860 8000 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]
