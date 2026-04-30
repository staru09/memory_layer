FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-secret env vars (secrets set via Cloud Run env vars)
ENV GEMINI_MODEL=gemini-2.5-flash
ENV GEMINI_EMBEDDING_MODEL=gemini-embedding-001
ENV PG_PORT=5432
ENV PG_USER=postgres
ENV PG_DB=postgres

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
