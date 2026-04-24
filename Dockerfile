FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF processing and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLP model data
RUN python -m nltk.downloader stopwords punkt
RUN python -m spacy download pt_core_news_sm

COPY . .

# Make the entrypoint script executable
RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
