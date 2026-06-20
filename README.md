# PS·AI — PlayStation D2C AI Agent

A RAG-powered conversational AI agent that helps PlayStation players discover games, get personalized recommendations, and find answers — grounded in real game data from RAWG and IGDB.

Built to mirror the technical stack of Sony Interactive Entertainment's Direct to Consumer ML Engineering team.

---

## How It Works

```
Player Question
      ↓
FastAPI REST Endpoint
      ↓
LangGraph Agent
      ↓              ↓
Voyage AI         pgvector
Embeddings        HNSW Search
      ↓              ↓
   Top-K Relevant Chunks
      ↓
Claude Sonnet 4.6
      ↓
Grounded Answer + LangSmith Trace
```

1. A player asks a natural language question
2. The LangGraph agent embeds the question using Voyage AI (`voyage-4`)
3. pgvector's HNSW index finds the most semantically similar game chunks
4. Retrieved context is injected into Claude's prompt
5. Claude generates a grounded answer — no hallucination, sources only
6. Every run is traced in LangSmith for observability

---

## Stack

| Layer         | Technology                             |
| ------------- | -------------------------------------- |
| Orchestration | LangGraph                              |
| LLM           | Anthropic Claude Sonnet 4.6            |
| Embeddings    | Voyage AI `voyage-4` (1024-dim)        |
| Vector Store  | PostgreSQL + pgvector (HNSW index)     |
| Data Sources  | RAWG API + IGDB API (Twitch OAuth 2.0) |
| API           | FastAPI + uvicorn                      |
| Observability | LangSmith                              |
| Frontend      | React + Vite + Tailwind CSS            |
| Deployment    | Azure Container Instances              |

---

## Project Structure

```
ps-d2c-agent/
├── src/
│   ├── config.py              # Environment variable loader + validation
│   ├── ingestion/
│   │   ├── rawg_fetcher.py    # RAWG API client
│   │   ├── igdb_fetcher.py    # IGDB API client (OAuth 2.0)
│   │   ├── chunker.py         # Semantic chunking strategy
│   │   ├── embedder.py        # Voyage AI embedding pipeline
│   │   ├── vector_store.py    # pgvector operations (setup, store, search)
│   │   └── pipeline.py        # Full ingestion orchestration
│   ├── retrieval/
│   │   └── retriever.py       # RAG retrieval + context formatting
│   ├── agent/
│   │   └── graph.py           # LangGraph agent (retrieve → generate)
│   └── api/
│       └── main.py            # FastAPI app + routes
├── frontend/                  # React + Vite + Tailwind UI
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## Data Pipeline

Game data is ingested from two complementary sources:

- **RAWG** — 200 top-rated PlayStation games (PS3/PS4/PS5) with ratings, release dates, playtime, genres, and platform availability
- **IGDB** — 100 top-rated PlayStation games with summaries, developer/publisher info, and community ratings

Each game is split into focused semantic chunks:

```
God of War
├── [overview]   — name, rating, release date, playtime/summary
├── [genres]     — Action, Adventure
└── [companies]  — SIE Santa Monica Studio, Sony Interactive Entertainment
```

**897 total chunks** stored in pgvector with 1024-dimensional Voyage AI embeddings and an HNSW cosine similarity index.

---

## RAG Design Decisions

**Why semantic chunking over full-document embedding?**
Splitting each game into focused chunks (overview, genres, companies) means a query about _"who made this game"_ hits the companies chunk directly at high similarity, rather than matching a diluted full-document embedding.

**Why Voyage AI over OpenAI embeddings?**
`voyage-4` uses asymmetric embedding — `input_type="document"` for indexing and `input_type="query"` for search — optimizing both sides of the retrieval independently. This improves retrieval precision without extra engineering cost.

**Why pgvector over a managed vector database?**
PostgreSQL is already standard in production stacks. pgvector adds vector search without a new infrastructure dependency, simplifying compliance, access control, and operational overhead.

**Why LangGraph over a simple chain?**
State is explicit and inspectable at every step. Adding reranking, tool calling, or human-review nodes requires one new node and one new edge — not a full rewrite.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (for pgvector)
- Node.js 18+ (for frontend)

### 1. Clone the repo

```bash
git clone git@github.com:Ayush-20122/ps-d2c-agent.git
cd ps-d2c-agent
```

### 2. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 3. Set up pgvector

```bash
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=ps_agent \
  -p 5433:5432 \
  ankane/pgvector
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```
ANTHROPIC_API_KEY=
RAWG_API_KEY=
IGDB_CLIENT_ID=
IGDB_CLIENT_SECRET=
VOYAGE_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=ps-d2c-agent
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://aws.api.smith.langchain.com
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/ps_agent
```

### 5. Run the ingestion pipeline

```bash
python src/ingestion/pipeline.py
```

This fetches 300 games, chunks them into 897 semantic vectors, embeds them with Voyage AI, and stores them in pgvector. Run once.

### 6. Start the backend

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`

---

## API

### `GET /health`

```json
{ "status": "healthy", "service": "PlayStation D2C AI Agent" }
```

### `POST /ask`

```json
// Request
{ "question": "What are good PS5 exclusives?" }

// Response
{
  "question": "What are good PS5 exclusives?",
  "answer": "Based on the available data, Astro Bot (90.54/100) is a standout PS5 exclusive..."
}
```

---

## Observability

Every agent run is traced in LangSmith showing:

- Retrieved chunks and similarity scores
- Exact prompt sent to Claude
- Token usage and cost per run
- Latency breakdown by node (retrieve → generate)

---

## Example Queries

```
"What are the best PS5 exclusives right now?"
"Who developed God of War Ragnarök?"
"Recommend an open world RPG with a great story"
"Games similar to Dark Souls on PlayStation"
"What is the highest rated PS4 game?"
```

---

## API Keys Required

| Service                                    | Purpose       | Free Tier        |
| ------------------------------------------ | ------------- | ---------------- |
| [Anthropic](https://console.anthropic.com) | Claude LLM    | Pay per token    |
| [RAWG](https://rawg.io/apidocs)            | Game database | Free             |
| [IGDB](https://dev.twitch.tv)              | Game database | Free             |
| [Voyage AI](https://dash.voyageai.com)     | Embeddings    | 200M tokens free |
| [LangSmith](https://smith.langchain.com)   | Observability | Free tier        |

---

## Author

**Ayush Bhattacharyya**
AI Engineer
[GitHub](https://github.com/Ayush-20122)
