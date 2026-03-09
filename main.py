#!/usr/bin/env python3
"""Minimal LangGraph workflow with real OpenAI token streaming."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from config import AppConfig, load_config


class AgentState(TypedDict, total=False):
    user_input: str
    intent: Literal["time", "chat"]
    tool_result: str
    response: str


def _emit(payload: dict[str, Any]) -> None:
    try:
        get_stream_writer()(payload)
    except RuntimeError:
        pass


def _chunk_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _build_chat_model(config: AppConfig):
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "api_key": config.openai_api_key,
        "model": config.openai_chat_model,
        "temperature": 0,
    }
    if config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    return ChatOpenAI(**kwargs)


def _build_graph(config: AppConfig):
    model = _build_chat_model(config)
    graph = StateGraph(AgentState)

    def classify_intent(state: AgentState) -> AgentState:
        intent: Literal["time", "chat"] = (
            "time" if "time" in state["user_input"].strip().lower() else "chat"
        )
        _emit({"event": "node", "node": "classify_intent", "intent": intent})
        return {"intent": intent}

    def run_basic_tool(state: AgentState) -> AgentState:
        if state["intent"] != "time":
            return {"tool_result": ""}
        tool_result = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _emit({"event": "tool", "tool": "get_utc_time", "output": tool_result})
        return {"tool_result": tool_result}

    def build_response(state: AgentState) -> AgentState:
        _emit(
            {
                "event": "model",
                "provider": "openai",
                "model": config.openai_chat_model,
            }
        )

        tool_result = state.get("tool_result") or ""
        prompt = (
            f"User message: {state['user_input']}\n"
            + (
                f"Tool result (UTC): {tool_result}\nUse tool result directly."
                if tool_result
                else "No tool call needed."
            )
        )
        messages = [
            SystemMessage(content="Reply briefly and clearly."),
            HumanMessage(content=prompt),
        ]

        parts: list[str] = []
        for chunk in model.stream(messages):
            text = _chunk_text(chunk.content)
            if not text:
                continue
            parts.append(text)
            _emit({"event": "token", "text": text})
        return {"response": "".join(parts).strip()}

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("run_basic_tool", run_basic_tool)
    graph.add_node("build_response", build_response)
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "run_basic_tool")
    graph.add_edge("run_basic_tool", "build_response")
    graph.add_edge("build_response", END)
    return graph.compile()


CONFIG = load_config()
GRAPH = _build_graph(CONFIG)


def stream_agent_events(user_input: str):
    for mode, chunk in GRAPH.stream(
        {"user_input": user_input}, stream_mode=["updates", "custom"]
    ):
        yield {"mode": mode, "data": chunk}


def run_cli(user_input: str) -> str:
    final_response = ""
    print("\nStreaming events:\n")
    for event in stream_agent_events(user_input):
        mode = event["mode"]
        data = event["data"]
        if mode == "updates":
            for node_name, node_update in data.items():
                print(f"[update:{node_name}] {node_update}")
                if "response" in node_update:
                    final_response = node_update["response"]
        elif mode == "custom":
            if data.get("event") == "token":
                print(data["text"], end="", flush=True)
            else:
                print(f"\n[custom] {data}")
    print("\n")
    return final_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple LangGraph workflow with real OpenAI streaming."
    )
    parser.add_argument(
        "--input",
        default="hello world",
        help='Input text for the graph (default: "hello world").',
    )
    args = parser.parse_args()
    final = run_cli(args.input)
    print("Final response:")
    print(final)
