FROM python:3.12-slim

WORKDIR /app

COPY agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY tools/ tools/
COPY agent/ agent/

CMD ["python", "-m", "agent.worker"]
