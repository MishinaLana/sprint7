FROM python:3.9.23 as base

WORKDIR /app

# RUN pip install --upgrade pip

RUN pip install faiss-cpu \
    langchain \
    sentence-transformers \
    yandex_chain

COPY . .

CMD ["python", "main.py"]

