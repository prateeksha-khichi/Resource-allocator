FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 7860
ENV PYTHONPATH=/app
ENV DATABASE_URL=""
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""
CMD bash -c "uvicorn api.routes:app --host 0.0.0.0 --port 8000 & streamlit run dashboard.py --server.port 7860 --server.address 0.0.0.0"
