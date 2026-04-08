from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware   # ✅ ADD THIS

from app.logging_config import setup_logging
from app.models import ChatQuery, FeedbackRequest
from app.orchestrator import SentimentSystem

system = SentimentSystem()


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    yield


app = FastAPI(
    title="Indian Market Sentiment Bot",
    version="1.0.0",
    lifespan=lifespan,
)

# ✅ ADD THIS BLOCK HERE (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest")
async def ingest_data() -> dict:
    try:
        return await system.ingest()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
async def chat(query: ChatQuery) -> dict:
    try:
        response = await system.ask(query.question)
        return response.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/feedback")
async def feedback(request: FeedbackRequest) -> dict:
    try:
        return await system.feedback(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc