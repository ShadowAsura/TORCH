FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_MODULE=torch_net.scheduler_app:app \
    PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home torch && chown -R torch:torch /app
USER torch

EXPOSE 8000

# Compose builds the worker image separately from Dockerfile.worker.
CMD ["sh", "-c", "exec uvicorn \"$APP_MODULE\" --host 0.0.0.0 --port \"$PORT\""]
