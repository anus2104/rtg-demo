FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "\
from rtg_hybrid_rag.indexer import IndexBuilder; \
from rtg_hybrid_rag.settings import settings; \
IndexBuilder(settings).build_or_load()"

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
