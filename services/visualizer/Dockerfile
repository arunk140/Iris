FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-cache

COPY visualizer.py ./

FROM base AS runner
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "visualizer.py", "--server.port=8501", "--server.address=0.0.0.0"]
