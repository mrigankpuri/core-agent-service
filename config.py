#!/usr/bin/env python3
"""Minimal app config."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    openai_api_key: str = Field(min_length=1)
    openai_chat_model: str = Field(default="gpt-4.1-mini", min_length=1)
    openai_base_url: str | None = None
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_config() -> AppConfig:
    _load_env_file(Path(__file__).resolve().parent / ".env.local")
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=os.getenv("API_PORT", "8000"),
    )
