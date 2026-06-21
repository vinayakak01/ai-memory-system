# Memory Agent

<img width="1918" height="1078" alt="Memory Agent UI" src="https://github.com/user-attachments/assets/872515a8-92b1-4e83-9d1c-5f1e62e6ead4" />


Memory Agent is a production-minded ChatGPT-style memory system. It pairs a polished conversational UI with a selective long-term memory layer, so the assistant can remember durable user preferences, projects, and goals across turns instead of treating every message as stateless.

The project is intentionally small enough to run locally, but structured like a serious service:

- FastAPI application with package-based boundaries
- local Ollama inference for chat and memory extraction
- SQLite persistence with concurrency-safe configuration
- sentence-transformers embeddings for memory retrieval
- modern web UI with live memory inspection and controls

## Why this project exists

Most chatbot demos store everything or remember nothing. This project sits in the middle:

- it stores only information likely to improve future conversations
- it retrieves only the highest-value memories for the current turn
- it keeps the memory system inspectable through a visible sidebar
- it degrades gracefully when memory persistence or the local model is unavailable

That makes it a strong foundation for a future multi-user or cloud-hosted memory platform.

## Core capabilities

- Personalized chat backed by retrieved long-term memory
- Automatic memory extraction from conversation turns
- Typed memories: `semantic`, `episodic`, and `procedural`
- Manual memory controls: add, remember latest, delete
- Local inference with `ollama` and configurable model selection
- SQLite write-retry and WAL-mode configuration for better local concurrency

## Architecture at a glance

### Write path

`message -> extract candidate memories -> evaluate -> store`

After each assistant reply, the system attempts to extract durable memories from the latest exchange. It keeps only high-signal facts such as:

- response preferences
- ongoing projects
- long-term goals
- durable user background facts

### Read path

`new message -> retrieve memories -> rank -> compose context -> answer`

On each new turn, the system retrieves memory candidates using embeddings and ranks them with:

- semantic similarity
- recency
- importance

Only the top memories are injected into the LLM prompt to keep the context compact and relevant.

## Repository layout

```text
memory-agent-mvp/
├── app/
│   ├── core/
│   │   ├── database.py
│   │   └── settings.py
│   ├── db/
│   │   ├── models.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── llm_client.py
│   │   └── memory_service.py
│   ├── web/
│   │   ├── static/
│   │   │   ├── app.js
│   │   │   └── styles.css
│   │   └── templates/
│   │       └── index.html
│   └── main.py
├── data/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Technology stack

- Backend: FastAPI, SQLAlchemy, Pydantic
- LLM runtime: Ollama
- Default model: `llama3.2:3b`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Storage: SQLite
- Frontend: server-rendered HTML, CSS, and JavaScript

## Local setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create `.env` from `.env.example`:

```env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://127.0.0.1:11434
APP_TITLE=Memory Agent MVP
```

### 3. Start Ollama

```bash
ollama pull llama3.2:3b
ollama serve
```

### 4. Run the application

```bash
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## API surface

- `GET /api/health`
- `GET /api/memories`
- `POST /api/chat`
- `POST /api/memories`
- `POST /api/memories/remember-latest`
- `DELETE /api/memories/{id}`

## Reliability notes

This MVP includes a few important production-minded behaviors:

- SQLite runs with `WAL` mode, `busy_timeout`, and retry handling for lock contention.
- Chat continues even if memory writes fail temporarily.
- If Ollama is unavailable, the API reports degraded health and returns a clear `503`.
- If structured memory extraction fails, the app falls back to a lightweight heuristic extractor.

## Demo flow

Try this sequence:

1. `I'm building a memory system MVP.`
2. `I prefer concise technical explanations.`
3. `I'm interested in RAG systems and knowledge distillation.`
4. `How should I plan the next version of my project?`

You should see:

- new memories appear in the sidebar
- the assistant respond with more relevant context on the fourth turn
- the memory list remain editable through manual add and delete actions
