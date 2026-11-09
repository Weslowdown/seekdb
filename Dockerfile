FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY server.py /app/server.py
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN mkdir -p /data

WORKDIR /data
ENTRYPOINT ["python3", "/app/server.py", "--data-dir=/data"]
