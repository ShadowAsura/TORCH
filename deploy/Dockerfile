# One image for every TORCH node; each container picks its node with --node.
# CPU only: jobs are simulated, so no GPU runtime is needed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN python -m pip install --no-cache-dir "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "httpx>=0.27,<1"

COPY torchsched ./torchsched
COPY deploy/*.json ./deploy/

RUN useradd --create-home torch
USER torch
EXPOSE 8000
ENTRYPOINT ["python", "-m", "torchsched.runtime.http", "--host", "0.0.0.0", "--port", "8000"]
