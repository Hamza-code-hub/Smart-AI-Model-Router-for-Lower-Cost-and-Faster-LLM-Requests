FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

COPY src/ src/
COPY alembic.ini .
COPY migrations/ migrations/
COPY config/ config/
COPY scripts/ scripts/

EXPOSE 8000

CMD alembic upgrade head && uvicorn gateway.main:app --host 0.0.0.0 --port 8000
