FROM python:3.12-slim

WORKDIR /app

# Install orchestrator dependencies
COPY orchestrator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules from repo root
COPY shared/ shared/

# Copy orchestrator service code
COPY orchestrator/ orchestrator/

CMD ["python", "-m", "orchestrator.submit", "--help"]
