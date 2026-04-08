# Indian Stock Sentiment Multi-Agent Backend

Production-ready Python backend for grounded Indian stock sentiment analysis over:
- NSE/BSE-style bulk & block deals (last 24h, NSE endpoints + archival fallback)
- 100 recent Indian trading videos from YouTube via Apify

The system is designed as a modular multi-agent architecture using a Hermes-compatible orchestration loop, OpenRouter LLM, vector retrieval, strict citation enforcement, and a closed feedback loop.

## Architecture

### Agents
1. **Market Data Agent**
   - Pulls last 24h bulk/block deal records from NSE JSON endpoints
   - Normalizes fields (symbol, side, quantity, price, client)
   - Fallback to NSE archive CSV endpoints when needed

2. **YouTube Scraper Agent**
   - Uses Apify actor (`APIFY_YOUTUBE_ACTOR_ID`)
   - Fetches 100 newest videos matching `YOUTUBE_QUERY`
   - Extracts transcript/captions + metadata (channel, title, timestamp, URL)
   - Transcript sources (in priority order):
     - Apify subtitles segments (timestamped)
     - Apify transcript field (if present)
     - YouTube Transcript API (`youtube-transcript-api`) (timestamped)
     - Timestamped description topics (parsed from description)
     - Raw description text (last resort)

3. **Sentiment Analysis Agent**
   - Runs transcript-level sentiment extraction via OpenRouter
   - Extracts stock symbols and sentiment mentions
   - Aggregates symbol-level bullish/bearish/neutral counts

4. **RAG Agent / Knowledge Base**
   - Chunks transcript text with overlap
   - Embeds chunks with `sentence-transformers`
   - Stores vectors in FAISS and metadata in SQLite
   - Supports semantic retrieval for grounded answering

5. **Chatbot Agent**
   - Retrieves relevant evidence first
   - Generates answer *only* from retrieved context
   - States that sentiment is aggregated from multiple recent trading videos
   - Adds an explicit trading recommendation based on bullish/bearish/neutral output
   - Returns explicit citations (channel + timestamp + snippet)
   - Fails safely when evidence is insufficient

6. **Learning Loop Agent**
   - Captures user feedback and rating
   - Stores interactions in SQLite
   - Tunes retrieval depth (`top_k`) as a closed improvement loop

### Hermes Integration
- `app/orchestrator.py` includes a Hermes runtime bridge (`HermesRuntime`) that wraps agent execution in a loop hook.
- If `hermes-agent` import fails, the orchestrator logs fallback behavior and keeps pipeline operational.

## Chunking Strategy (RAG)

- If timestamped segments exist (from Apify subtitles / YouTube Transcript API / description topics), each segment is indexed as its own document for precise citation.
- Otherwise, normalize whitespace, then fixed-size character chunks (`~700`) with overlap (`~120`).
- Overlap preserves context continuity across sentence boundaries.
- Each chunk is stored with source metadata:
  - YouTube: `video_id`, `channel`, `timestamp` (real when available, otherwise estimated)
  - Deals: `deal_type`, `symbol`, `trade_date`, client/price/qty

This supports high-recall retrieval while preserving citation traceability.

## Folder Structure

```text
.
├── .env.example
├── requirements.txt
├── README.md
├── main.py
└── app
    ├── __init__.py
    ├── api.py
    ├── cli.py
    ├── config.py
    ├── logging_config.py
    ├── models.py
    ├── orchestrator.py
    ├── agents
    │   ├── __init__.py
    │   ├── base.py
    │   ├── market_agent.py
    │   ├── youtube_agent.py
    │   ├── sentiment_agent.py
    │   ├── rag_agent.py
    │   ├── chatbot_agent.py
    │   └── learning_agent.py
    ├── data
    │   ├── __init__.py
    │   ├── market_data.py
    │   └── youtube_data.py
    ├── services
    │   ├── __init__.py
    │   ├── chunker.py
    │   ├── openrouter_client.py
    │   ├── storage.py
    │   ├── symbols.py
    │   └── vector_store.py
    └── utils
        ├── __init__.py
        └── retry.py
```

## Setup

1. Create virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment:

```bash
copy .env.example .env
```

Update `.env` with valid keys:
- `OPENROUTER_API_KEY`
- `APIFY_TOKEN`

3. Run API server:

```bash
python main.py
```

## API Usage

### 1) Ingest last-24h data
```bash
curl -X POST http://localhost:8000/ingest
```

### 2) Ask grounded sentiment question
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What’s sentiment on RELIANCE?\"}"
```

### 3) Send feedback (learning loop)
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What’s sentiment on RELIANCE?\",\"answer\":\"...\",\"rating\":4,\"feedback_text\":\"Good but include FII context\"}"
```

## CLI Usage

```bash
python -m app.cli ingest
python -m app.cli ask --question "Why bullish on HDFC?"
python -m app.cli feedback --question "What are FIIs doing?" --answer "..." --rating 3 --feedback-text "Need stronger citations"
```

## Sample Input / Output

### Sample question
`What’s sentiment on RELIANCE?`

### Sample response shape
```json
{
  "answer": "Stock Market Sentiment: BULLISH\n\nSentiment is based on aggregated signals from multiple recent trading videos.\n\n🧠 Key Insights:\n- RELIANCE is repeatedly discussed as strong above key support levels\n- Multiple recent transcript snippets mention continued upside momentum\n\n📈 Recommendation:\nConsider buying opportunities with caution.\n\n📌 Sources:\n1. Nifty Next Breakout Watchlist (Trader Alpha @ 00:01:30, BUY)\n2. Top Swing Trades This Week (Market Mentor @ 00:03:10, BUY)",
  "citations": [
    {
      "source_type": "youtube",
      "source_id": "yt:abc123:2",
      "title": "Nifty Next Breakout Watchlist",
      "channel": "Trader Alpha",
      "timestamp": "00:01:30",
      "published_at": null,
      "snippet": "RELIANCE looks strong above..."
    }
  ],
  "grounded": true
}
```

## Production Notes

- Retries with exponential backoff for external APIs (`tenacity`)
- Async I/O for ingestion and model calls (`httpx`, `async/await`)
- SQLite + FAISS persistence for local production deployment
- Chat response caching (SQLite):
  - Repeated questions return cached responses without calling OpenRouter again
  - Cache TTL controlled by `CHAT_CACHE_TTL_SECONDS`
  - Cache is keyed by normalized question + model + retrieval depth + corpus fingerprint
- Strong grounding policy:
  - Retrieval-first pipeline
  - Citation IDs must map to retrieved context
  - If no valid citations: answer returns non-grounded fallback
- Structured logging and explicit error handling on API routes

## Scalability Improvements (Next)

- Replace SQLite with Postgres + pgvector for distributed retrieval
- Add task queue (Celery/RQ) for scheduled ingestion every 15-30 min
- Add Redis cache for hot queries and deduplicated transcript processing
- Use batched embedding workers and background indexing jobs
- Add observability (OpenTelemetry traces + Prometheus metrics)
