import json

from app.agents.base import BaseAgent
from app.config import settings
from app.models import ChatResponse, SourceCitation
from app.services.openrouter_client import OpenRouterClient


class ChatbotAgent(BaseAgent):
    def __init__(self, llm: OpenRouterClient) -> None:
        self.llm = llm

    async def run(self, payload: dict) -> ChatResponse:
        question: str = payload["question"]
        context_docs: list[dict] = payload["context_docs"]
        if not context_docs:
            return ChatResponse(
                answer=(
                    "Sentiment: neutral\n"
                    "Reasoning: Insufficient transcript evidence in the last 24h dataset.\n"
                    "Citations: none"
                ),
                citations=[],
                grounded=False,
            )
        context = [
            {
                "id": d["id"],
                "title": d["title"],
                "channel": d.get("channel"),
                "timestamp": d.get("timestamp"),
                "snippet": d["snippet"],
            }
            for d in context_docs
        ]
        response = await self.llm.chat(
            system_prompt=(
                "You are a grounded stock sentiment analyst. "
                "Use only provided context. No external facts. "
                "If context is insufficient, explicitly say insufficient evidence. "
                "Return strict JSON: "
                "{sentiment:bullish|bearish|neutral,reasoning:string,citation_ids:string[]}. "
                "Reasoning must be based on transcript/deal snippets only."
            ),
            user_prompt=f"Question: {question}\n\nContext:\n{json.dumps(context)}",
            temperature=0.0,
        )
        citation_ids = response.get("citation_ids", [])
        by_id = {doc["id"]: doc for doc in context_docs}
        selected: list[dict] = []
        seen: set[str] = set()
        for cid in citation_ids:
            sid = str(cid)
            if sid in seen:
                continue
            doc = by_id.get(sid)
            if not doc:
                continue
            selected.append(doc)
            seen.add(sid)
        grounded = len(selected) > 0
        citations = [
            SourceCitation(
                source_type=doc["source_type"],
                source_id=doc["id"],
                title=doc["title"],
                channel=doc.get("channel"),
                timestamp=doc.get("timestamp"),
                published_at=None,
                snippet=doc["snippet"],
            )
            for doc in selected[: settings.top_k_retrieval]
        ]
        sentiment = str(response.get("sentiment", "neutral")).lower()
        if sentiment not in {"bullish", "bearish", "neutral"}:
            sentiment = "neutral"
        reasoning = str(response.get("reasoning", "")).strip()
        if not grounded:
            reasoning = "Insufficient transcript evidence with valid citations from the latest dataset."
        insight_lines = []
        if reasoning:
            # Split into concise bullet-like insights.
            for part in reasoning.replace("\n", " ").split(". "):
                cleaned = part.strip().rstrip(".")
                if cleaned:
                    insight_lines.append(f"- {cleaned}")
                if len(insight_lines) >= 4:
                    break
        if not insight_lines:
            insight_lines = ["- No strong evidence available from recent transcripts."]

        citation_lines = []
        for cite in citations[:5]:
            channel = cite.channel or "Unknown channel"
            ts = cite.timestamp or "timestamp unavailable"
            title = cite.title or "Untitled source"
            side = _infer_trade_side(cite.snippet)
            side_label = f", {side}" if side else ""
            citation_lines.append(
                f"{len(citation_lines)+1}. {title} ({channel} @ {ts}{side_label})"
            )

        sentiment_label = sentiment.upper()
        if sentiment_label == "BULLISH":
            recommendation = "Consider buying opportunities with caution."
        elif sentiment_label == "BEARISH":
            recommendation = "Consider selling or avoiding short-term positions."
        else:
            recommendation = "Hold or wait for clearer signals."

        answer = (
            f"Stock Market Sentiment: {sentiment_label}\n\n"
            "Sentiment is based on aggregated signals from multiple recent trading videos.\n\n"
            " Key Insights:\n"
            + "\n".join(insight_lines)
            + "\n\n Recommendation:\n"
            + recommendation
            + "\n\n Sources:\n"
            + ("\n".join(citation_lines) if citation_lines else "1. No grounded sources available")
        )
        return ChatResponse(answer=answer, citations=citations, grounded=grounded)


def _infer_trade_side(snippet: str) -> str:
    text = (snippet or "").lower()
    if " sell" in f" {text} ":
        return "SELL"
    if " buy" in f" {text} ":
        return "BUY"
    return ""
