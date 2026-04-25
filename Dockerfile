FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

EXPOSE 7860

# Health check for HuggingFace Spaces
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "ecocloud_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
