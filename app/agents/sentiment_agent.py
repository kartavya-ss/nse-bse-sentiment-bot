from collections import defaultdict

from app.agents.base import BaseAgent
from app.models import SentimentAggregate, SentimentMention, TranscriptRecord
from app.services.openrouter_client import OpenRouterClient
from app.services.symbols import extract_symbols


class SentimentAnalysisAgent(BaseAgent):
    def __init__(self, llm: OpenRouterClient) -> None:
        self.llm = llm

    async def run(self, payload: dict) -> dict:
        transcripts: list[TranscriptRecord] = payload["transcripts"]
        mentions: list[SentimentMention] = []
        for record in transcripts:
            source_text = record.transcript
            symbols = extract_symbols(source_text)
            if not symbols:
                continue
            response = await self.llm.chat(
                system_prompt=(
                    "You are a strict financial sentiment extractor. "
                    "Only classify sentiment that is explicitly stated in the provided source text "
                    "(either transcript or timestamped description topics). "
                    "Return JSON: {mentions:[{symbol,sentiment,confidence,rationale}]}"
                ),
                user_prompt=(
                    f"Source text:\n{source_text[:5000]}\n\n"
                    f"Allowed symbols: {sorted(symbols)}"
                ),
            )
            for item in response.get("mentions", []):
                symbol = str(item.get("symbol", "")).upper()
                if symbol not in symbols:
                    continue
                sentiment = item.get("sentiment", "neutral")
                if sentiment not in {"bullish", "bearish", "neutral"}:
                    sentiment = "neutral"
                mentions.append(
                    SentimentMention(
                        symbol=symbol,
                        sentiment=sentiment,
                        confidence=float(item.get("confidence", 0.5)),
                        rationale=str(item.get("rationale", ""))[:300],
                        source_id=record.video_id,
                    )
                )
        aggregate = defaultdict(lambda: SentimentAggregate(symbol=""))
        for mention in mentions:
            if not aggregate[mention.symbol].symbol:
                aggregate[mention.symbol] = SentimentAggregate(symbol=mention.symbol)
            bucket = aggregate[mention.symbol]
            if mention.sentiment == "bullish":
                bucket.bullish += 1
            elif mention.sentiment == "bearish":
                bucket.bearish += 1
            else:
                bucket.neutral += 1
        return {"mentions": mentions, "aggregate": list(aggregate.values())}
