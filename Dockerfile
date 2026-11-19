FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir .
RUN python scripts/download_data.py && trendpredictor train
EXPOSE 8000
CMD ["uvicorn", "trendpredictor.api:app", "--host", "0.0.0.0", "--port", "8000"]
