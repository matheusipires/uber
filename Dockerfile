FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# deps básicos p/ compilar rodas se necessário
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip moderno evita vários erros de wheel
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# copia o projeto
COPY . .

# cria pasta do volume
RUN mkdir -p /data

# Streamlit deve escutar no $PORT fornecido pelo Render
ENV HIST_DB_PATH=/data/history.sqlite
EXPOSE 8000
CMD streamlit run app.py --server.port ${PORT:-8000} --server.address 0.0.0.0

