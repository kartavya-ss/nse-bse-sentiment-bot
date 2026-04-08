from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    source_type: Literal["youtube", "market_deal"]
    source_id: str
    title: str
    channel: str | None = None
    timestamp: str | None = None
    published_at: datetime | None = None
    snippet: str


class TranscriptRecord(BaseModel):
    video_id: str
    title: str
    channel: str
    published_at: datetime
    transcript: str
    url: str
    transcript_source: Literal["apify_subtitles", "apify_transcript", "youtube_api", "description"] = "apify_subtitles"
    topic_segments: list[dict[str, str]] = Field(default_factory=list)


class MarketDealRecord(BaseModel):
    symbol: str
    deal_type: Literal["bulk", "block"]
    trade_date: datetime
    client_name: str | None = None
    buy_sell: str | None = None
    quantity: float | None = None
    price: float | None = None
    value_lakhs: float | None = None
    raw: dict = Field(default_factory=dict)


class SentimentMention(BaseModel):
    symbol: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: float
    rationale: str
    source_id: str


class SentimentAggregate(BaseModel):
    symbol: str
    bullish: int = 0
    bearish: int = 0
    neutral: int = 0

    @property
    def dominant(self) -> str:
        distribution = {
            "bullish": self.bullish,
            "bearish": self.bearish,
            "neutral": self.neutral,
        }
        return max(distribution, key=distribution.get)


class ChatQuery(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[SourceCitation]
    grounded: bool


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(ge=1, le=5)
    feedback_text: str | None = None
