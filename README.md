#  NSE/BSE Sentiment Analysis Chatbot

An AI-powered chatbot that analyzes Indian stock market sentiment using **YouTube trading insights + market deal data**, built with **FastAPI, RAG (Retrieval-Augmented Generation), and OpenRouter LLMs**.

---

##  Overview

This project solves the problem of **fragmented market insights** by aggregating:

* 📺 YouTube trading videos (via Apify)
* 📈 NSE/BSE bulk & block deal data
* 🤖 AI-based sentiment analysis

 The chatbot provides:

* Market sentiment (Bullish / Bearish / Neutral)
* Key insights from real data
* Source-backed answers (no hallucination)
* Trading recommendations

---

##  Key Features

*  **RAG-based chatbot** (grounded responses)
*  **Multi-source data aggregation**
*  **Sentiment analysis engine**
*  **Trading recommendations**
*  **Source citations (channel + timestamp)**
*  **No hallucination (grounded = true/false)**
*  **Caching system (cost optimized)**
*  **Rate limit handling + fallback system**

---

##  Architecture

```
User Query
    ↓
FastAPI Backend
    ↓
RAG Retrieval (Vector DB)
    ↓
Context + Market Data
    ↓
OpenRouter LLM
    ↓
Structured Response (Sentiment + Insights + Sources)
```

---

##  Tech Stack

* **Backend:** FastAPI (Python)
* **LLM:** OpenRouter (Mistral / LLaMA)
* **Scraping:** Apify (YouTube data)
* **Embeddings:** Sentence Transformers
* **Vector DB:** FAISS
* **Database:** SQLite
* **Architecture:** Agent-based (Hermes-inspired)

---

##  Project Structure

```
app/
 ├── agents/          # AI agents (chatbot, sentiment, RAG, etc.)
 ├── services/        # OpenRouter, vector store, storage
 ├── data/            # Data ingestion logic
 ├── models.py        # Pydantic schemas
 ├── api.py           # FastAPI endpoints

main.py               # Entry point
.env                  # API keys (not committed)
```

---

##  Setup Instructions

### 1️ Clone the repository

```
git clone https://github.com/kartavya-ss/nse-bse-sentiment-bot.git
cd nse-bse-sentiment-bot
```

---

### 2️ Install dependencies

```
pip install -r requirements.txt
```

---

### 3️ Create `.env` file

```
OPENROUTER_API_KEY=your_key_here
APIFY_TOKEN=your_token_here
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
```

---

### 4️ Run the server

```
uvicorn app.api:app --reload
```

---

### 5️ Open API docs

```
http://127.0.0.1:8000/docs
```

---

##  Usage

### Step 1: Ingest Data (run once)

```
POST /ingest
```

 Fetches YouTube + market data and stores in vector DB

---

### Step 2: Ask Questions

```
POST /chat
```

Example:

```json
{
  "question": "What is sentiment on Nifty?"
}
```

---

##  Example Output

```
 Market Sentiment: BEARISH

 Key Insights:
- EXIMROUTES bulk deal SELL indicates institutional selling
- Mixed BUY/SELL signals observed
- Sentiment is derived from aggregated market activity

 Sources:
1. EXIMROUTES bulk deal (NSE, SELL)
2. ATALREAL bulk deal (NSE, BUY)

 Recommendation:
Consider avoiding short-term positions due to selling pressure.
```

---

##  Design Decisions

*  **RAG used to avoid hallucination**
*  **Caching implemented to reduce API cost**
*  **Fallback system ensures response even if LLM fails**
*  **Strict schema validation using Pydantic**

---

##  Important Notes

* `.env` is excluded for security
* API keys should never be committed
* Ingest step should not be run repeatedly (cost control)

---

##  Future Improvements

* Real-time market data integration
* Frontend dashboard (React)
* Advanced financial indicators
* Multi-language support

---

##  Author

**Kartavya Agarwal**

---

##  Acknowledgment

Built as part of an AI internship task focusing on:

* Agent-based systems
* RAG architecture
* Financial sentiment analysis
