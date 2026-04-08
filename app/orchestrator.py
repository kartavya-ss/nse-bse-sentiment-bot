import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.agents.chatbot_agent import ChatbotAgent
from app.agents.learning_agent import LearningLoopAgent
from app.agents.market_agent import MarketDataAgent
from app.agents.rag_agent import RAGAgent
from app.agents.sentiment_agent import SentimentAnalysisAgent
from app.agents.youtube_agent import YouTubeScraperAgent
from app.config import settings
from app.data.market_data import MarketDataClient
from app.data.youtube_data import YouTubeApifyClient
from app.models import ChatResponse, FeedbackRequest, SourceCitation
from app.services.openrouter_client import OpenRouterClient
from app.services.storage import SQLiteStore
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HermesRuntime:
    """
    Thin Hermes bridge to keep the project aligned with Hermes Agent framework.
    """

    def __init__(self) -> None:
        self.available = False
        try:
            import hermes_agent  # type: ignore

            _ = hermes_agent
            self.available = True
        except Exception:
            logger.warning("hermes-agent import failed; using local orchestration fallback")

    async def loop(self, name: str, payload: dict[str, Any], fn) -> Any:
        return await fn(payload)


class SentimentSystem:
    def __init__(self) -> None:
        data_root = Path(settings.data_dir)
        self.store = SQLiteStore(data_root / "knowledge.db")
        self.vector_store = VectorStore(data_root / "vectors")
        self.llm = OpenRouterClient()
        self.hermes = HermesRuntime()

        self.market_agent = MarketDataAgent(MarketDataClient())
        self.youtube_agent = YouTubeScraperAgent(YouTubeApifyClient())
        self.sentiment_agent = SentimentAnalysisAgent(self.llm)
        self.rag_agent = RAGAgent(self.store, self.vector_store)
        self.chat_agent = ChatbotAgent(self.llm)
        self.learning_agent = LearningLoopAgent(self.store)

        self.dynamic_top_k = settings.top_k_retrieval

    async def ingest(self) -> dict:
        market_task = self.hermes.loop("market_data", {}, self.market_agent.run)
        youtube_task = self.hermes.loop("youtube_scrape", {"limit": 100}, self.youtube_agent.run)

        market_deals, transcripts = await asyncio.gather(market_task, youtube_task)

        sentiment = await self.hermes.loop(
            "sentiment_analysis",
            {"transcripts": transcripts},
            self.sentiment_agent.run,
        )

        indexed = await self.hermes.loop(
            "rag_indexing",
            {"transcripts": transcripts, "market_deals": market_deals},
            self.rag_agent.run,
        )

        return {
            "market_deals": len(market_deals),
            "transcripts": len(transcripts),
            "sentiment_symbols": len(sentiment["aggregate"]),
            "indexed_documents": indexed["indexed_documents"],
        }

    async def ask(self, question: str) -> ChatResponse:
        # 🔑 Normalize question
        normalized = " ".join(question.strip().lower().split())

        # 🔑 Cache key
        response_format_version = "v3"
        corpus_fingerprint = str(getattr(self.vector_store.index, "ntotal", 0))
        raw_key = (
            f"{response_format_version}|{settings.openrouter_model}|"
            f"{self.dynamic_top_k}|{corpus_fingerprint}|{normalized}"
        )
        cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

        #  Check cache first
        cached = self.store.get_chat_cache(cache_key, ttl_seconds=settings.chat_cache_ttl_seconds)
        if cached:
            try:
                return ChatResponse.model_validate(cached)
            except ValidationError:
                logger.warning("Invalid cached ChatResponse detected; invalidating cache key")
                self.store.delete_chat_cache(cache_key)

        #  Retrieve limited docs (important for cost + rate limit)
        docs = await self.rag_agent.retrieve(question, top_k=min(self.dynamic_top_k, 3))
        if not docs:
            return ChatResponse(
                answer="No relevant data found for this query.",
                citations=[],
                grounded=False,
            )

        try:
            #  Delay to avoid rate limit
            await asyncio.sleep(2)

            #  Single LLM call
            response = await self.hermes.loop(
                "chatbot_answer",
                {"question": question, "context_docs": docs},
                self.chat_agent.run,
            )

        except Exception as e:
            #  Fallback if LLM fails (429 or any error)
            logger.warning(f"LLM failed, using fallback: {e}")

            fallback_text = " **Market Sentiment:** "

            # Slightly more robust fallback sentiment logic from grounded snippets.
            buy_count = 0
            sell_count = 0
            for d in docs:
                text = str(d.get("snippet", "")).lower()
                if "sell" in text:
                    sell_count += 1
                elif "buy" in text:
                    buy_count += 1
            if sell_count > buy_count:
                sentiment = "BEARISH"
            elif buy_count > sell_count:
                sentiment = "BULLISH"
            else:
                sentiment = "NEUTRAL"
            fallback_text += sentiment + "\n\n Key Insights:\n"
            citations: list[SourceCitation] = []
            unique_sources: dict[str, dict[str, Any]] = {}
            for d in docs:
                key = (
                    str(d.get("source_id") or d.get("id") or "").strip()
                    or f"{d.get('title','')}|{d.get('channel','')}|{d.get('timestamp','')}"
                )
                if key not in unique_sources:
                    unique_sources[key] = d

            unique_docs = list(unique_sources.values())[:3]

            seen: set[str] = set()
            for d in unique_docs:
                snippet = str(d.get("snippet", "")).strip()
                if snippet and snippet not in seen:
                    fallback_text += f"- {snippet}\n"
                    seen.add(snippet)

            # Add deterministic summary line from inferred sentiment.
            if sentiment == "BEARISH":
                fallback_text += "- Institutional selling indicates negative short-term sentiment\n"
            elif sentiment == "BULLISH":
                fallback_text += "- Institutional buying indicates positive market outlook\n"

            fallback_text += (
                "\nSentiment is based on aggregated signals from multiple recent trading videos.\n"
            )
            if sentiment == "BULLISH":
                recommendation = "Consider buying opportunities with caution."
            elif sentiment == "BEARISH":
                recommendation = "Consider selling or avoiding short-term positions."
            else:
                recommendation = "Hold or wait for clearer signals."
            fallback_text += f"\n Recommendation:\n{recommendation}\n\n Sources:\n"
            for i, d in enumerate(unique_docs, 1):
                source_type = d.get("source_type", "youtube")
                if source_type not in {"youtube", "market_deal"}:
                    source_type = "youtube"
                title = d.get("title", "Unknown")
                citation = SourceCitation(
                    source_type=source_type,
                    source_id=d.get("id", f"fallback:{i}"),
                    title=title,
                    channel=d.get("channel"),
                    timestamp=d.get("timestamp"),
                    published_at=None,
                    snippet=d.get("snippet", "") or title,
                )
                citations.append(citation)
                channel = citation.channel or "Unknown channel"
                ts = citation.timestamp or "timestamp unavailable"
                side = _infer_trade_side(citation.snippet)
                side_label = f", {side}" if side else ""
                fallback_text += f"{i}. {title} ({channel} @ {ts}{side_label})\n"

            response = ChatResponse(
                answer=fallback_text.strip(),
                citations=citations,
                grounded=True,
            )

        #  Cache result
        self.store.upsert_chat_cache(cache_key, response.model_dump())

        return response

    async def feedback(self, request: FeedbackRequest) -> dict:
        outcome = await self.hermes.loop(
            "learning_loop",
            request.model_dump(),
            self.learning_agent.run,
        )
        self.dynamic_top_k = outcome["recommended_top_k"]
        return outcome


def _infer_trade_side(snippet: str) -> str:
    text = (snippet or "").lower()
    if " sell" in f" {text} ":
        return "SELL"
    if " buy" in f" {text} ":
        return "BUY"
    return ""
