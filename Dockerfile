# Base completa (não-slim) para evitar dor com wheels nativas
FROM python:3.11-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# libs de runtime e build que às vezes fazem falta p/ pandas/altair/etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl ca-certificates git \
    libstdc++6 libgomp1 tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip moderno evita vários erros de wheel
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary -r requirements.txt

# copia o projeto
COPY . .

# diretório do volume persistente
RUN mkdir -p /data

# APP: aponta SQLite para o disco /data
ENV HIST_DB_PATH=/data/history.sqlite

# Render injeta $PORT; local uso 8000
EXPOSE 8000
CMD streamlit run app.py --server.port ${PORT:-8000} --server.address 0.0.0.0
