FROM ghcr.io/astral-sh/uv:0.12.10 AS uv
FROM python:3.12-slim-bookworm AS base
COPY --from=uv /uv /bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=2 \
    NUMBA_NUM_THREADS=2 \
    ARTIFACT_DIR=/app/artifacts \
    USER_DB=/app/state/users.sqlite3 \
    PATH="/app/.venv/bin:$PATH"

FROM base AS trainer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsuitesparse-dev && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --group train --group sansa --group dev --no-cache
COPY movie_recs ./movie_recs
COPY tests ./tests
CMD ["python", "-m", "movie_recs.train"]

FROM base AS runtime
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-group train --no-group sansa --no-cache
COPY movie_recs ./movie_recs
COPY artifacts/*.npz artifacts/*.json artifacts/DATASET_LICENSE.txt ./artifacts/
RUN mkdir /app/state && chown 10001:10001 /app/state
USER 10001:10001
VOLUME ["/app/state"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["uvicorn", "movie_recs.api:app", "--host", "0.0.0.0", "--port", "8000"]
