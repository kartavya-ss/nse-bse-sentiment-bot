from app.agents.base import BaseAgent
from app.data.market_data import MarketDataClient
from app.models import MarketDealRecord


class MarketDataAgent(BaseAgent):
    def __init__(self, client: MarketDataClient) -> None:
        self.client = client

    async def run(self, payload: dict | None = None) -> list[MarketDealRecord]:
        return await self.client.fetch_last_24h_deals()
