# Serving image: the API and the exported model, no training or MLflow.
# Build for linux/amd64 (what Azure Container Apps runs).

FROM python:3.13-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim
# LightGBM needs the OpenMP runtime, which the slim image does not have.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ARTIFACTS_DIR=/app/artifacts
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY loan_default/ loan_default/
COPY artifacts/model.joblib artifacts/metadata.json artifacts/
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "loan_default.api:app", "--host", "0.0.0.0", "--port", "8000"]
