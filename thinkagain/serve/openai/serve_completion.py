"""FastAPI wiring that exposes ThinkAgain graphs via the OpenAI API surface."""

import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
except ImportError:
    raise ImportError(
        "FastAPI is required for the OpenAI server. "
        "Install with: pip install -e '.[serve]' (or: pip install 'thinkagain[serve]')"
    )

try:
    from jsonschema import ValidationError, validate as jsonschema_validate
except ImportError:
    jsonschema_validate = None  # type: ignore[assignment]
    ValidationError = None  # type: ignore[assignment]

from thinkagain import Context, Graph

from .models import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    DeltaMessage,
    JsonSchema,
    Message,
    ResponseFormat,
    Usage,
)

NormalizedResponseFormat = Optional[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


class GraphRegistry:
    """Track graphs and the default entry."""

    def __init__(self):
        self._graphs: Dict[str, Graph] = {}
        self._default: Optional[str] = None

    def register(self, name: str, graph: Graph, set_default: bool = False):
        self._graphs[name] = graph
        if set_default or not self._default:
            self._default = name

    def get(self, name: Optional[str] = None) -> tuple[Graph, str]:
        """Get graph by name or default. Returns (graph, name) tuple."""
        key = name or self._default
        if not key:
            raise KeyError("No default graph set in registry")
        if key not in self._graphs:
            raise KeyError(f"Graph '{key}' not found in registry")
        return self._graphs[key], key

    def list_graphs(self) -> List[str]:
        return list(self._graphs.keys())


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


def _openai_error(
    message: str,
    *,
    type_: str = "server_error",
    param: Optional[str] = None,
    status_code: int = 500,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {"message": message, "type": type_, "param": param, "code": None}
        },
    )


def _invalid_request(message: str, *, param: Optional[str] = None) -> HTTPException:
    return _openai_error(
        message, type_="invalid_request_error", param=param, status_code=400
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _read_field(obj: Any, field: str) -> Any:
    """Read field from dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4 if text else 0


def _get_usage(result: Any, user_query: str, response_text: str) -> Usage:
    """Extract or estimate token usage from the graph result."""
    usage = _read_field(result, "usage")

    if hasattr(usage, "to_usage"):
        return usage.to_usage()

    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
    else:
        prompt, completion = (
            _estimate_tokens(user_query),
            _estimate_tokens(response_text),
        )

    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _model_fingerprint(model: str) -> str:
    return f"fp_{hashlib.md5(model.encode()).hexdigest()[:12]}"


def _extract_user_query(messages: List[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            return msg.content
    return ""


def _stringify_response(raw: Any) -> str:
    if raw is None:
        return "No response generated"
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw)
    except TypeError:
        return str(raw)


def _resolve_graph(registry: GraphRegistry, model: Optional[str]) -> tuple[Graph, str]:
    """Fetch graph for the model name, raising OpenAI-style error if missing."""
    try:
        return registry.get(model)
    except KeyError as exc:
        raise _openai_error(
            str(exc), type_="invalid_request_error", param="model", status_code=404
        ) from exc


def _build_context(
    request: ChatCompletionRequest,
    model: str,
    user_query: str,
    response_format: NormalizedResponseFormat,
) -> Context:
    ctx = Context(
        **request.model_dump(exclude={"model", "stream"}),
        model=model,
        user_query=user_query,
    )
    if response_format:
        ctx.response_format = response_format
    return ctx


# -----------------------------------------------------------------------------
# Response Format Validation
# -----------------------------------------------------------------------------


def _normalize_response_format(
    rf: Optional[ResponseFormat],
) -> NormalizedResponseFormat:
    if not rf or rf.type == "text":
        return None
    if rf.type == "json_object":
        return {"type": "json_object"}
    if rf.type == "json_schema":
        return _validate_and_normalize_schema(rf.json_schema)
    raise _invalid_request(
        f"Unsupported response_format type '{rf.type}'", param="response_format.type"
    )


def _validate_and_normalize_schema(schema: Optional[JsonSchema]) -> Dict[str, Any]:
    if not schema or not schema.schema_:
        raise _invalid_request(
            "response_format.json_schema.schema is required when type is 'json_schema'.",
            param="response_format",
        )

    if jsonschema_validate is None:
        raise _openai_error(
            "Structured outputs require 'jsonschema'. Install with: pip install 'thinkagain[serve]'",
            status_code=500,
        )

    body = schema.schema_
    if schema.strict is not True:
        raise _invalid_request(
            "Structured outputs require json_schema.strict = true.",
            param="response_format.json_schema.strict",
        )
    if body.get("type") != "object":
        raise _invalid_request(
            "Structured outputs require a root object schema.",
            param="response_format.json_schema.schema.type",
        )
    if body.get("additionalProperties") is not False:
        raise _invalid_request(
            "Structured output schemas must set additionalProperties: false.",
            param="response_format.json_schema.schema.additionalProperties",
        )

    props, req = body.get("properties"), body.get("required")
    if not isinstance(props, dict) or not props:
        raise _invalid_request(
            "Structured output schemas must define properties.",
            param="response_format.json_schema.schema.properties",
        )
    if not isinstance(req, list) or set(props) != set(req):
        raise _invalid_request(
            "Structured outputs require every property to be listed as required.",
            param="response_format.json_schema.schema.required",
        )

    return {
        "type": "json_schema",
        "json_schema": schema.model_dump(by_alias=True, exclude_none=True),
    }


def _validate_structured_response(text: str, fmt: NormalizedResponseFormat):
    if not fmt:
        return

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _openai_error(
            f"Model response is not valid JSON: {exc}",
            type_="invalid_response_error",
            status_code=422,
        )

    if fmt["type"] == "json_object":
        if not isinstance(parsed, dict):
            raise _openai_error(
                "JSON mode responses must serialize to an object.",
                type_="invalid_response_error",
                status_code=422,
            )
        return

    schema_body = fmt.get("json_schema", {}).get("schema")
    if not schema_body:
        raise _openai_error(
            "Structured output schema missing at validation time.", status_code=500
        )

    try:
        jsonschema_validate(instance=parsed, schema=schema_body)
    except ValidationError as exc:  # type: ignore[misc]
        raise _openai_error(
            f"Model response does not match schema: {exc.message}",
            type_="invalid_response_error",
            status_code=422,
        )


# -----------------------------------------------------------------------------
# Response Building
# -----------------------------------------------------------------------------


def _build_response(
    request_id: str,
    model: str,
    result: Any,
    user_query: str,
    response_format: NormalizedResponseFormat,
) -> ChatCompletionResponse:
    raw, refusal = _read_field(result, "response"), _read_field(result, "refusal")
    text = _stringify_response(raw)

    if not refusal:
        _validate_structured_response(text, response_format)

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(
                    role="assistant", content=None if refusal else text, refusal=refusal
                ),
                finish_reason="content_filter" if refusal else "stop",
            )
        ],
        usage=_get_usage(result, user_query, refusal or text),
        system_fingerprint=_model_fingerprint(model),
    )


# -----------------------------------------------------------------------------
# Streaming
# -----------------------------------------------------------------------------


@dataclass
class _StreamState:
    """Track streamed text and compute deltas."""

    response: str = ""
    refusal: str = ""

    def delta(self, field: str, current: Optional[str]) -> Optional[str]:
        cur = current or ""
        prev = getattr(self, field)
        if not cur or cur == prev:
            return None
        setattr(self, field, cur)
        return cur[len(prev) :]


def _stream_chunk(
    request_id: str,
    created: int,
    model: str,
    fingerprint: str,
    *,
    content: Optional[str] = None,
    refusal: Optional[str] = None,
    role: Optional[str] = None,
    finish: Optional[str] = None,
) -> str:
    chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(role=role, content=content, refusal=refusal),
                finish_reason=finish,
            )
        ],
        system_fingerprint=fingerprint,
    )
    return f"data: {chunk.model_dump_json()}\n\n"


async def _stream_response(
    graph: Graph,
    ctx: Context,
    request_id: str,
    model: str,
    response_format: NormalizedResponseFormat = None,
) -> AsyncIterator[str]:
    created, fingerprint = int(time.time()), _model_fingerprint(model)
    yield _stream_chunk(
        request_id, created, model, fingerprint, role="assistant", content=""
    )

    state = _StreamState()
    async for event in graph.stream(ctx):
        if event.type != "node" or not event.streaming:
            continue
        for field in ("refusal", "response"):
            if delta := state.delta(field, getattr(event.ctx, field, None)):
                yield _stream_chunk(
                    request_id,
                    created,
                    model,
                    fingerprint,
                    **{"refusal" if field == "refusal" else "content": delta},
                )
                break

    if response_format and not state.refusal:
        _validate_structured_response(state.response, response_format)

    yield _stream_chunk(
        request_id,
        created,
        model,
        fingerprint,
        finish="content_filter" if state.refusal else "stop",
    )
    yield "data: [DONE]\n\n"


# -----------------------------------------------------------------------------
# App Factory
# -----------------------------------------------------------------------------


def create_app(registry: GraphRegistry) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        print(f"\n{'=' * 72}\nThinkAgain OpenAI-Compatible Server\n{'=' * 72}")
        print(f"Registered graphs: {registry.list_graphs()}\n{'=' * 72}\n")
        yield

    app = FastAPI(
        title="ThinkAgain OpenAI API",
        description="OpenAI-compatible API for ThinkAgain graph execution",
        lifespan=lifespan,
    )

    @app.get("/")
    async def root():
        return {
            "message": "ThinkAgain OpenAI-Compatible API Server",
            "endpoints": {
                "chat_completions": "/v1/chat/completions",
                "models": "/v1/models",
                "health": "/health",
            },
            "graphs": registry.list_graphs(),
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/v1/models")
    async def list_models():
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": n, "object": "model", "created": now, "owned_by": "thinkagain"}
                for n in registry.list_graphs()
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        user_query = _extract_user_query(request.messages)
        if not user_query:
            raise _invalid_request("No user message found", param="messages")

        graph, model = _resolve_graph(registry, request.model)
        fmt = _normalize_response_format(request.response_format)
        ctx = _build_context(request, model, user_query, fmt)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        try:
            if request.stream:
                return StreamingResponse(
                    _stream_response(graph, ctx, request_id, model, fmt),
                    media_type="text/event-stream",
                )
            return _build_response(
                request_id, model, await graph.arun(ctx), user_query, fmt
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise _openai_error(
                f"Unexpected server error: {exc}", status_code=500
            ) from exc

    return app
