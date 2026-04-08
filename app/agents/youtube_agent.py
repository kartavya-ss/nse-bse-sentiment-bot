from app.agents.base import BaseAgent
from app.data.youtube_data import YouTubeApifyClient
from app.models import TranscriptRecord


class YouTubeScraperAgent(BaseAgent):
    def __init__(self, client: YouTubeApifyClient) -> None:
        self.client = client

    async def run(self, payload: dict | None = None) -> list[TranscriptRecord]:
        limit = 100 if payload is None else int(payload.get("limit", 100))
        return await self.client.fetch_recent_trading_videos(limit=limit)
