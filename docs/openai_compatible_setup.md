# OpenAI-Compatible Provider Setup

The original project used Azure OpenAI environment variables. This refactor also supports OpenAI-compatible chat providers.

## 1. Configure `.env`

Copy:

```bash
cp .env.example.openai-compatible .env
```

Then fill:

```text
MODEL_PROVIDER=openai_compatible
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_API_KEY=your-key
OPENAI_MODEL_NAME=gpt-5.5
```

Do not commit `.env`.

## 2. Embeddings

Live RAG ingestion and retrieval also need an embedding model:

```text
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Use a model that your provider actually supports. If the provider only offers chat models, the offline demo and routing eval still work, but live RAG vector search will not be fully available.

For local demos where the gateway does not provide embeddings, use the deterministic local hashing fallback:

```text
USE_LOCAL_HASHING_EMBEDDINGS=true
EMBEDDING_DIM=1536
```

This keeps FAQ ingestion and local retrieval runnable without sending embedding requests to the provider. It is a demo fallback, not a production medical embedding strategy.

## 3. Ingest FAQ Knowledge

The project includes a 150-entry patient FAQ layer under `data/faq/`. Ingest it with:

```bash
python ingest_rag_data.py --faq-dir data/faq
```

Each FAQ becomes one searchable chunk with metadata such as domain, priority, risk level, doctor-needed flag, source organization, and source URL.

## 4. Run Offline Checks

These do not require model calls:

```bash
python demo/offline_agentic_walkthrough.py
python scripts/run_local_checks.py demo intent_eval tool_eval
```

## 5. Run Full App

After installing dependencies and configuring `.env`:

```bash
pip install -r requirements.txt
python app.py
```
