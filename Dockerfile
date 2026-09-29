# sensorlab governance dashboard + scoring API
#
#   docker build -t sensorlab .
#   docker run --rm -p 8000:8000 sensorlab                       # demo model trained at build time
#   docker run --rm -p 8000:8000 -v $PWD/models:/app/models sensorlab   # your own models/pipeline.joblib
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[serve]"

# Train the reference model into the image so the container is self-contained.
# Mount /app/models to serve a model trained elsewhere instead.
# The image ships the published multi-seed results next to a single-seed reference model.
COPY artifacts/results.json ./artifacts/results.json
RUN sensorlab train --seed 0 --model-out models/pipeline.joblib

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD ["sensorlab", "serve", "--host", "0.0.0.0", "--port", "8000"]
