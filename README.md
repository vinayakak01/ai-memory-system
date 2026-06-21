# Memory Agent

<img width="1918" height="1078" alt="image" src="https://github.com/user-attachments/assets/872515a8-92b1-4e83-9d1c-5f1e62e6ead4" />


A minimal but polished ChatGPT-style memory assistant built with FastAPI, SQLite, sentence-transformers, and local Ollama.

## What it does

- chat with a personalized assistant
- extract durable memories from conversation
- show memories live in a sidebar
- retrieve relevant memories on future turns
- add explicit memories manually
- remember the latest user message with one click
- delete memories

## Tech stack

- FastAPI
- SQLite + SQLAlchemy
- `sentence-transformers/all-MiniLM-L6-v2`
- local Ollama via the `ollama` Python library
- custom HTML/CSS/JS frontend

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and set:

```env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://127.0.0.1:11434
```

4. Run the app:

```bash
uvicorn main:app --reload
```

5. Make sure Ollama is running and the model is available:

```bash
ollama pull llama3.2:3b
ollama serve
```

6. Open `http://127.0.0.1:8000`

## API routes

- `GET /api/health`
- `GET /api/memories`
- `POST /api/chat`
- `POST /api/memories`
- `POST /api/memories/remember-latest`
- `DELETE /api/memories/{id}`

## Notes

- Retrieval ranking is intentionally simple: semantic similarity + recency + importance.
- If Ollama is unavailable, chat returns a clear `503` error and `/api/health` reports degraded status.
- If automatic memory extraction fails, the app falls back to a small heuristic extractor.
- This MVP is single-user and local-only by design.
