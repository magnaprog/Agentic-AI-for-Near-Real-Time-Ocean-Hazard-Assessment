FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN pip install --no-cache-dir "psycopg[binary]>=3.2.0,<4"

COPY src/hazard_assessment /app/hazard_assessment

# This is the only container handed DB_ADMIN_PASSWORD, and it needs no root
# capability: it opens a TCP connection and applies SQL. Drop to the same
# unprivileged user the core image uses.
RUN groupadd --system appuser && useradd --system --gid appuser appuser
USER appuser

ENTRYPOINT ["python", "-m", "hazard_assessment.storage.provision", "--max-attempts", "60", "--retry-delay-sec", "2"]
