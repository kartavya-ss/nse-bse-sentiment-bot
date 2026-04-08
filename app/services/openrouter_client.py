import json
from typing import Any

import httpx

from app.config import settings
from app.utils.retry import with_retry


class OpenRouterClient:
    def __init__(self) -> None:
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

    @with_retry(max_attempts=3)
    async def chat(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.1
    ) -> dict[str, Any]:
        payload = {
            "model": settings.openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
        message = data["choices"][0]["message"]["content"]
        return json.loads(message)
