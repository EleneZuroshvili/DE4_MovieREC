# Movie Recommendation AI Agent — Implementation Plan

## Overview

A **Movie Recommendation Chatbot** built with OpenAI Agents SDK + AWS Bedrock, demoed in Chainlit.

**Capabilities used (3 of 6):**
- **RAG** — ChromaDB movie database with semantic search
- **Multi-agent orchestration** — Orchestrator delegates to specialist sub-agents
- **Memory** — SQLiteSession for multi-turn conversation history

---

## Architecture

```
User (Chainlit UI)
        │
        ▼
┌──────────────────────────────┐
│   Movie Advisor Agent        │  ← orchestrator (Claude 3 Haiku)
│   (routes to sub-agents)     │
│   + Memory (SQLiteSession)   │
└──────┬──────────┬────────────┘
       │          │
       ▼          ▼
┌────────────┐  ┌──────────────────┐
│ Movie DB   │  │ Profile Builder  │
│ Agent      │  │ Agent            │
│ (RAG)      │  │ (preference      │
│ Nova Lite  │  │  extraction)     │
└────────────┘  │ Nova Lite        │
                └──────────────────┘
```

### Agent Roles

| Agent | Model | Job |
|-------|-------|-----|
| **Movie Advisor** (orchestrator) | Claude 3 Haiku | Routes requests, synthesizes final recommendation |
| **Profile Builder** | Nova Lite | Extracts genre, mood, audience preferences from user message |
| **Movie Database Agent** | Nova Lite | Queries ChromaDB via `movie_lookup_tool` for matching movies |

### Flow for a Recommendation Request

```
1. User: "I want a scary movie for date night"
2. Movie Advisor calls profile-builder tool
   → Profile Builder returns: "Horror/thriller for a couple, intense atmosphere"
3. Movie Advisor calls movie-database-search tool with that profile
   → Movie DB Agent queries ChromaDB, returns top 5 matches
4. Movie Advisor writes friendly response with 3-5 recommendations
```

### Memory (Multi-turn)

```
Turn 1: "I like sci-fi"           → recommends Interstellar, Blade Runner...
Turn 2: "something lighter"       → remembers sci-fi, suggests Galaxy Quest...
Turn 3: "I liked the second one"  → remembers recommendations, finds similar
```

---

## Files

```
chatbot/
├── create_movie_db.py    — Loads movies_metadata.csv into ChromaDB (run once)
├── movie_agent.py        — Agent definitions (3 agents + RAG tool)
├── movie_chatbot.py      — Chainlit app (streaming + memory)

data/
└── movies_metadata.csv   — TMDB movie dataset (~45k rows)

chroma/                   — ChromaDB persistent storage (generated)
```

### File Details

#### `create_movie_db.py` — ChromaDB Loader (run once)

- Reads `data/movies_metadata.csv` (TMDB dataset with 45k rows)
- **Parses JSON-like columns** — genres like `[{'id': 16, 'name': 'Animation'}, ...]` → `"Animation, Comedy, Family"` using `ast.literal_eval`
- **Filters for quality** — keeps only Released movies with overview and ≥10 votes
- Creates rich semantic documents per movie:
  ```
  Movie: Toy Story
  Year: 1995
  Genres: Animation, Comedy, Family
  Overview: Led by Woody, Andy's toys live happily...
  Rating: 7.7/10 (5415 votes)
  Runtime: 81 minutes
  ```
- Stores in ChromaDB collection `movie_db` with metadata (title, year, genres, rating, runtime, etc.)
- Batched inserts (500 per batch) to handle large dataset

#### `movie_agent.py` — Agent Definitions

- `bedrock_tool()` — wrapper to convert OpenAI Agents SDK tools to Bedrock-compatible format
- `movie_lookup_tool` — `@function_tool` that queries ChromaDB with semantic search
- `movie_db_agent` — RAG agent with the lookup tool (Nova Lite)
- `profile_builder_agent` — extracts user preferences into a searchable profile (Nova Lite)
- `movie_advisor` — orchestrator using sub-agents as tools via `as_tool()` (Claude 3 Haiku)

#### `movie_chatbot.py` — Chainlit App

- `@cl.on_chat_start` — initializes `SQLiteSession("movie_conversation_history")` for memory
- `@cl.on_message` — runs `movie_advisor` with streaming via `Runner.run_streamed()`
- Streams text deltas word-by-word to the UI
- Shows tool calls as Chainlit steps (e.g. "🔧 profile-builder", "🔧 movie-database-search")

---

## How to Run

### Step 1: Load movie data into ChromaDB (one time)
```bash
.venv/Scripts/python.exe chatbot/create_movie_db.py
```

### Step 2: Start the chatbot
```bash
chainlit run chatbot/movie_chatbot.py --port 10000
```

### Step 3: Open browser
```
http://localhost:10000
```

---

## Dataset Columns (movies_metadata.csv)

From TMDB. The loader uses these columns:

| Column | Used for |
|--------|----------|
| `title` | Display + search document |
| `overview` | Main search text (semantic search matches against this) |
| `genres` | Parsed from JSON-like list → "Action, Thriller" |
| `release_date` | Year extraction |
| `vote_average` | Rating display + metadata |
| `vote_count` | Quality filter (≥10 votes) |
| `runtime` | Display |
| `tagline` | Display |
| `original_language` | Metadata |
| `popularity` | Metadata |
| `production_companies` | Parsed from JSON-like list |
| `status` | Filter (only "Released") |
| `adult` | Metadata |

---

## Progress

- [x] Created `chatbot/create_movie_db.py` (ChromaDB loader)
- [x] Created `chatbot/movie_agent.py` (3 agents + RAG tool)
- [x] Created `chatbot/movie_chatbot.py` (Chainlit app + memory)
- [x] Fixed corrupted ChromaDB (deleted and recreated chroma/ folder)
- [x] Installed missing `pandas` dependency via `uv pip install pandas`
- [ ] Re-run `create_movie_db.py` to rebuild ChromaDB collection
- [ ] Start chatbot and test all 3 capabilities

---

## Test Scenarios

| Test | What it validates |
|------|-------------------|
| "Recommend a sci-fi movie" | RAG search + multi-agent (profile → search → response) |
| "Something more family-friendly" | Memory (remembers previous context) |
| "Tell me more about the first one" | Memory recall of specific recommendations |
| "I'm in the mood for a thriller, watching alone" | Profile builder extracts mood + audience |
| "I liked Inception, suggest similar" | RAG semantic search for similar movies |
