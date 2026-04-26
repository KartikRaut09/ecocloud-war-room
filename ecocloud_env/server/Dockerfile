FROM python:3.10-slim

WORKDIR /app

COPY ecocloud_env/server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "ecocloud_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]

