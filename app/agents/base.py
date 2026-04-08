from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    @abstractmethod
    async def run(self, payload: Any) -> Any:
        raise NotImplementedError
