"""LLM provider implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _openai_format_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tool schemas to OpenAI format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


@dataclass
class VLLM:
    """vLLM local model provider."""

    base_url: str = "http://localhost:8000/v1"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    api_key: str = "EMPTY"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate completion using vLLM OpenAI-compatible API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", 0.7),
            "max_tokens": kwargs.pop("max_tokens", 2048),
            **kwargs,
        }

        if tools:
            request_params["tools"] = _openai_format_tools(tools)

        response = await client.chat.completions.create(**request_params)
        message = response.choices[0].message

        result = {"role": "assistant", "content": message.content or ""}

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in message.tool_calls
            ]

        return result


@dataclass
class OpenAI:
    """OpenAI model provider."""

    model: str = "gpt-4"
    api_key: str | None = None
    base_url: str | None = None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate completion using OpenAI API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", 0.7),
            "max_tokens": kwargs.pop("max_tokens", 2048),
            **kwargs,
        }

        if tools:
            request_params["tools"] = _openai_format_tools(tools)

        response = await client.chat.completions.create(**request_params)
        message = response.choices[0].message

        result = {"role": "assistant", "content": message.content or ""}

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in message.tool_calls
            ]

        return result


@dataclass
class Anthropic:
    """Anthropic Claude model provider."""

    model: str = "claude-sonnet-4"
    api_key: str | None = None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate completion using Anthropic API."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        client = AsyncAnthropic(api_key=self.api_key)

        # Separate system messages from conversation
        system_messages = [m for m in messages if m.get("role") == "system"]
        non_system_messages = [m for m in messages if m.get("role") != "system"]

        request_params = {
            "model": self.model,
            "messages": non_system_messages,
            "max_tokens": kwargs.pop("max_tokens", 2048),
            "temperature": kwargs.pop("temperature", 0.7),
            **kwargs,
        }

        if system_messages:
            request_params["system"] = system_messages[0]["content"]

        if tools:
            request_params["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]

        response = await client.messages.create(**request_params)

        result = {"role": "assistant", "content": ""}
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {"id": block.id, "name": block.name, "arguments": block.input}
                )

        if tool_calls:
            result["tool_calls"] = tool_calls

        return result
