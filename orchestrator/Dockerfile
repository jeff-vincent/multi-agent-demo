FROM python:3.12-slim

WORKDIR /app

COPY orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY agent/ agent/
COPY orchestrator/ orchestrator/

CMD ["python", "-m", "orchestrator.submit", "--help"]
