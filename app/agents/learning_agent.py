from app.agents.base import BaseAgent
from app.services.storage import SQLiteStore


class LearningLoopAgent(BaseAgent):
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    async def run(self, payload: dict) -> dict:
        # Hermes-style closed loop: store explicit user rating and return operational guidance
        # that can be used by the orchestrator to tune retrieval depth and strictness.
        self.store.insert_feedback(
            question=payload["question"],
            answer=payload["answer"],
            rating=payload["rating"],
            feedback_text=payload.get("feedback_text"),
        )
        feedback = self.store.recent_feedback(limit=100)
        avg_rating = sum(item["rating"] for item in feedback) / max(len(feedback), 1)
        return {
            "feedback_count": len(feedback),
            "avg_rating": round(avg_rating, 2),
            "recommended_top_k": 10 if avg_rating < 3.5 else 8,
            "strict_grounding": avg_rating < 4.0,
        }
