from app.agents.base import BaseAgent
from app.models import MarketDealRecord, TranscriptRecord
from app.services.chunker import chunk_text
from app.services.storage import SQLiteStore
from app.services.vector_store import VectorStore


class RAGAgent(BaseAgent):
    def __init__(self, store: SQLiteStore, vectors: VectorStore) -> None:
        self.store = store
        self.vectors = vectors

    async def run(self, payload: dict) -> dict:
        transcripts: list[TranscriptRecord] = payload.get("transcripts", [])
        market_deals: list[MarketDealRecord] = payload.get("market_deals", [])
        doc_ids: list[str] = []
        chunks: list[str] = []

        for video in transcripts:
            if video.topic_segments:
                for idx, segment in enumerate(video.topic_segments):
                    snippet = segment.get("topic", "").strip()
                    if not snippet:
                        continue
                    source = video.transcript_source
                    doc_id = f"yt:{video.video_id}:{source}:{idx}"
                    self.store.upsert_document(
                        {
                            "id": doc_id,
                            "source_type": "youtube",
                            "title": video.title,
                            "channel": video.channel,
                            "timestamp": segment.get("timestamp") or None,
                            "published_at": video.published_at.isoformat(),
                            "snippet": snippet,
                            "metadata": {
                                "url": video.url,
                                "video_id": video.video_id,
                                "transcript_source": source,
                            },
                        }
                    )
                    doc_ids.append(doc_id)
                    chunks.append(snippet)
            else:
                for idx, chunk in enumerate(chunk_text(video.transcript)):
                    doc_id = f"yt:{video.video_id}:{idx}"
                    self.store.upsert_document(
                        {
                            "id": doc_id,
                            "source_type": "youtube",
                            "title": video.title,
                            "channel": video.channel,
                            "timestamp": _estimate_timestamp(idx),
                            "published_at": video.published_at.isoformat(),
                            "snippet": chunk,
                            "metadata": {
                                "url": video.url,
                                "video_id": video.video_id,
                                "transcript_source": video.transcript_source,
                            },
                        }
                    )
                    doc_ids.append(doc_id)
                    chunks.append(chunk)

        for idx, deal in enumerate(market_deals):
            doc_id = f"deal:{deal.deal_type}:{deal.symbol}:{idx}:{deal.trade_date.date().isoformat()}"
            snippet = (
                f"{deal.symbol} {deal.deal_type} deal by {deal.client_name or 'unknown'} "
                f"{deal.buy_sell or ''} qty={deal.quantity} price={deal.price}"
            ).strip()
            self.store.upsert_document(
                {
                    "id": doc_id,
                    "source_type": "market_deal",
                    "title": f"{deal.symbol} {deal.deal_type} deal",
                    "channel": "NSE",
                    "timestamp": deal.trade_date.isoformat(),
                    "published_at": deal.trade_date.isoformat(),
                    "snippet": snippet,
                    "metadata": deal.raw,
                }
            )
            doc_ids.append(doc_id)
            chunks.append(snippet)
        self.vectors.add_texts(doc_ids, chunks)
        return {"indexed_documents": len(doc_ids)}

    async def retrieve(self, question: str, top_k: int) -> list[dict]:
        # Pull extra candidates, then re-rank to prioritize YouTube transcript evidence.
        hits = self.vectors.search(question, top_k=max(top_k * 3, top_k))
        ids = [doc_id for doc_id, _ in hits]
        docs = self.store.get_documents_by_ids(ids)
        by_id = {doc["id"]: doc for doc in docs}
        ordered = []
        seen_ids: set[str] = set()
        for doc_id, score in hits:
            if doc_id in seen_ids:
                continue
            doc = by_id.get(doc_id)
            if not doc:
                continue
            seen_ids.add(doc_id)
            # Prefer YouTube transcript content for answering; keep deals for context.
            boost = 0.05 if doc.get("source_type") == "youtube" else 0.0
            doc["score"] = score + boost
            ordered.append(doc)
        ordered.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        # Ensure transcript content dominates the context set.
        youtube_docs = [d for d in ordered if d.get("source_type") == "youtube"]
        other_docs = [d for d in ordered if d.get("source_type") != "youtube"]
        merged = youtube_docs[:top_k] + other_docs[: max(0, top_k - len(youtube_docs[:top_k]))]
        return merged[:top_k]


def _estimate_timestamp(index: int) -> str:
    seconds = index * 45
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"
