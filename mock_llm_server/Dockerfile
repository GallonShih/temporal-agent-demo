FROM python:3.13-slim
RUN pip install --no-cache-dir "fastapi>=0.162" "uvicorn>=0.54"
WORKDIR /srv
COPY __init__.py app.py ./mock_llm_server/
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "mock_llm_server.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
