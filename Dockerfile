FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/
COPY artifacts/graphsage_pure.pt \
     artifacts/model_meta.pkl \
     artifacts/graph_meta.pkl \
     artifacts/explainer_reference.pkl \
     ./artifacts/

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn src.serving.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
