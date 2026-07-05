FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-cache

COPY watcher.py ./

FROM base AS runner
EXPOSE 8080
CMD ["uv", "run", "python", "watcher.py"]
