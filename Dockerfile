# PP/PE FTIR classifier web service (CPU only). Build: docker build -t pppe-ftir .  Run: docker run -p 8000:8000 pppe-ftir
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
# torch is only needed to re-train deep models, not to serve; drop it for a lean image
RUN grep -v -i -E "^(torch|xgboost|lightgbm|seaborn|matplotlib|openpyxl|pytest|httpx)==" requirements.txt > req_serve.txt && pip install --no-cache-dir -r req_serve.txt
COPY src ./src
COPY webapp ./webapp
COPY models ./models
EXPOSE 8000
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
