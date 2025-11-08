FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY scripts/ scripts/
COPY housing_linear.joblib housing_linear.joblib

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "scripts.session_3.api:app", "--host", "0.0.0.0", "--port", "8000"]


