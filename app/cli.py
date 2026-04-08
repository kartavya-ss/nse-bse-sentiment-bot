import argparse
import asyncio
import json

from app.logging_config import setup_logging
from app.models import FeedbackRequest
from app.orchestrator import SentimentSystem


async def _run(args: argparse.Namespace) -> None:
    setup_logging()
    system = SentimentSystem()
    if args.command == "ingest":
        result = await system.ingest()
        print(json.dumps(result, indent=2))
    elif args.command == "ask":
        result = await system.ask(args.question)
        print(json.dumps(result.model_dump(), indent=2, default=str))
    elif args.command == "feedback":
        payload = FeedbackRequest(
            question=args.question,
            answer=args.answer,
            rating=args.rating,
            feedback_text=args.feedback_text,
        )
        result = await system.feedback(payload)
        print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Indian market sentiment backend CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest")

    ask = sub.add_parser("ask")
    ask.add_argument("--question", required=True)

    fb = sub.add_parser("feedback")
    fb.add_argument("--question", required=True)
    fb.add_argument("--answer", required=True)
    fb.add_argument("--rating", required=True, type=int)
    fb.add_argument("--feedback-text", default=None)

    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
