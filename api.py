#!/usr/bin/env python3
"""FastAPI wrapper for the minimal LangGraph streaming demo."""

from __future__ import annotations

import json
from typing import Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import load_config
from main import stream_agent_events


app = FastAPI(title="Core Agent Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Keep wildcard origins valid in browsers. Set specific origins for credentials.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="User message")


def _to_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _stream_chat(message: str) -> Generator[str, None, None]:
    for item in stream_agent_events(message):
        yield _to_sse(item["mode"], item["data"])
    yield _to_sse("done", {"ok": True})


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/chat-stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _stream_chat(message),
        media_type="text/event-stream",
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run("api:app", host=config.api_host, port=config.api_port, reload=False)
