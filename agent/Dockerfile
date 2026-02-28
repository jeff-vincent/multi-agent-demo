FROM python:3.12-slim

WORKDIR /app

# Install agent dependencies
COPY agent/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules from repo root
COPY shared/ shared/

# Copy agent service code
COPY agent/ agent/

CMD ["python", "-m", "agent.worker"]
