"""
OpenAI-Compatible API Server for ThinkAgain
============================================

This module provides reusable components to create an OpenAI-compatible API
server that wraps ThinkAgain graph execution.

Usage:
    1. Import create_app and GraphRegistry
    2. Register your custom graphs
    3. Run the server

Example:
    from thinkagain import Graph, Worker, Context, END
    from thinkagain.serve.openai.serve_completion import create_app, GraphRegistry

    # Define your worker
    class MyWorker(Worker):
        def __call__(self, ctx: Context) -> Context:
            ctx.response = f"Response to: {ctx.user_query}"
            return ctx

    # Build your graph
    graph = Graph(name="my_graph")
    graph.add_node("worker", MyWorker())
    graph.add_edge("worker", END)

    # Create registry and register your graph
    registry = GraphRegistry()
    registry.register("my-model", graph, set_default=True)

    # Create the FastAPI app
    app = create_app(registry)

    # Run with: uvicorn my_module:app
"""

import time
import uuid
from typing import AsyncIterator, Optional, Dict, List
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI and pydantic are required for the OpenAI server. "
        "Install them with: pip install -e '.[serve]' (or: pip install 'thinkagain[serve]')"
    )

from thinkagain import Context, Worker, Graph, END


# -----------------------------------------------------------------------------
# Pydantic Models for OpenAI API Compatibility
# -----------------------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "thinkagain"
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


# -----------------------------------------------------------------------------
# ThinkAgain Graph Registry
# -----------------------------------------------------------------------------


class GraphRegistry:
    """Registry for managing multiple ThinkAgain graphs."""

    def __init__(self):
        self._graphs: Dict[str, Graph] = {}
        self._default_graph_name: Optional[str] = None

    def register(self, name: str, graph: Graph, set_default: bool = False):
        """Register a graph with a name."""
        self._graphs[name] = graph
        if set_default or not self._default_graph_name:
            self._default_graph_name = name

    def get(self, name: Optional[str] = None) -> Graph:
        """
        Get a graph by name, or the default graph if no name provided.

        This method preserves the original behavior for backward compatibility.
        Newer code should prefer ``get_required()`` and ``get_default()`` for
        clearer semantics.
        """
        if name is None:
            return self.get_default()
        return self.get_required(name)

        return self._graphs[name]

    def get_default(self) -> Graph:
        """
        Get the default graph.

        Raises:
            KeyError: If no default graph has been registered.
        """
        if self._default_graph_name is None:
            raise KeyError("No default graph set in registry")
        return self._graphs[self._default_graph_name]

    def get_required(self, name: str) -> Graph:
        """
        Get a graph by name.

        Raises:
            KeyError: If the graph name does not exist.
        """
        if name not in self._graphs:
            raise KeyError(f"Graph '{name}' not found in registry")
        return self._graphs[name]

    def list_graphs(self) -> List[str]:
        """List all registered graph names."""
        return list(self._graphs.keys())


# Note: Don't use a global registry - users should create their own
# See create_app() function below for the proper pattern


# -----------------------------------------------------------------------------
# Example ThinkAgain Workers (Customizable)
# -----------------------------------------------------------------------------


class ProcessQuery(Worker):
    """Process the user's query and extract key information."""

    async def arun(self, ctx: Context) -> Context:
        # Extract the user query from messages
        user_message = ctx.user_query
        ctx.processed_query = user_message.strip()
        ctx.log(f"[{self.name}] Processed query: {ctx.processed_query}")
        return ctx


class GenerateResponse(Worker):
    """Generate a response based on the processed query."""

    async def arun(self, ctx: Context) -> Context:
        query = ctx.get("processed_query", ctx.user_query)

        # This is a mock response - replace with actual LLM call
        # Example: ctx.response = your_llm_client.complete(query)
        ctx.response = (
            f"I received your message: '{query}'\n\n"
            "This is a mock response from ThinkAgain. To integrate with an actual LLM:\n"
            "1. Replace this worker with your LLM API call\n"
            "2. Or pass an LLM client to the worker during initialization\n"
            "3. The response will be returned to the OpenAI-compatible API"
        )

        ctx.log(f"[{self.name}] Generated response ({len(ctx.response)} chars)")
        return ctx


class EnhanceResponse(Worker):
    """Optionally enhance or refine the response."""

    async def arun(self, ctx: Context) -> Context:
        # Add some enhancement logic
        original = ctx.response
        ctx.response = f"{original}\n\n---\nProcessed by ThinkAgain Graph"
        ctx.log(f"[{self.name}] Enhanced response")
        return ctx


def build_default_graph() -> Graph:
    """Build the default chat completion graph."""
    graph = Graph(name="chat_completion")

    # Add nodes
    graph.add_node("process", ProcessQuery())
    graph.add_node("generate", GenerateResponse())
    graph.add_node("enhance", EnhanceResponse())

    # Set up flow
    graph.set_entry("process")
    graph.add_edge("process", "generate")
    graph.add_edge("generate", "enhance")
    graph.add_edge("enhance", END)

    return graph


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def extract_user_query(messages: List[Message]) -> str:
    """Extract the last user message from the conversation."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4


async def execute_graph_streaming(
    graph: Graph, ctx: Context, request_id: str, model: str
) -> AsyncIterator[str]:
    """Execute graph with true incremental streaming support."""
    created = int(time.time())

    # Send initial chunk with role
    initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Execute the graph and stream incremental updates
    prev_response = ""
    async for event in graph.stream(ctx):
        # Stream intermediate updates from streaming workers
        if event.type == "node" and event.streaming and hasattr(event.ctx, "response"):
            current_response = event.ctx.response
            # Calculate delta (new content since last update)
            if current_response and current_response != prev_response:
                delta_content = current_response[len(prev_response) :]
                prev_response = current_response

                chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=DeltaMessage(content=delta_content),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

    # Send final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0, delta=DeltaMessage(), finish_reason="stop"
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# -----------------------------------------------------------------------------
# Error helpers (OpenAI-style shape)
# -----------------------------------------------------------------------------


def _openai_error(
    message: str,
    *,
    type_: str = "server_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    status_code: int = 500,
) -> HTTPException:
    """
    Return an HTTPException with an OpenAI-compatible error body.

    This keeps client behavior predictable for users of official SDKs.
    """
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": type_,
                "param": param,
                "code": code,
            }
        },
    )


# -----------------------------------------------------------------------------
# FastAPI Application Factory
# -----------------------------------------------------------------------------


def create_app(registry: GraphRegistry) -> FastAPI:
    """
    Create a FastAPI application with the given graph registry.

    Args:
        registry: GraphRegistry with your graphs already registered

    Returns:
        FastAPI application ready to run

    Example:
        registry = GraphRegistry()
        registry.register("my-model", my_graph, set_default=True)
        app = create_app(registry)
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize the application on startup."""
        print("\n" + "=" * 72)
        print("ThinkAgain OpenAI-Compatible Server")
        print("=" * 72)
        print(f"Registered graphs: {registry.list_graphs()}")
        print("=" * 72 + "\n")
        yield

    app = FastAPI(
        title="ThinkAgain OpenAI API",
        description="OpenAI-compatible API for ThinkAgain graph execution",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
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
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/v1/models")
    async def list_models():
        """List available models (graphs)."""
        graphs = registry.list_graphs()
        return {
            "object": "list",
            "data": [
                {
                    "id": graph_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "thinkagain",
                }
                for graph_name in graphs
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """
        OpenAI-compatible chat completions endpoint.

        This endpoint executes a ThinkAgain graph and returns the result
        in OpenAI's chat completion format.
        """
        # Basic validation: messages and user content
        user_query = extract_user_query(request.messages)
        if not user_query:
            raise _openai_error(
                "No user message found",
                type_="invalid_request_error",
                param="messages",
                status_code=400,
            )

        # Model selection: require explicit model name if provided
        try:
            model_name = request.model
            if model_name:
                graph = registry.get_required(model_name)
            else:
                graph = registry.get_default()
        except KeyError as exc:
            raise _openai_error(
                str(exc),
                type_="invalid_request_error",
                param="model",
                status_code=404,
            )

        # Create context with the user query and request metadata
        ctx = Context(
            user_query=user_query,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            n=request.n,
            stop=request.stop,
            model=model_name,
        )

        # Generate request ID
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Handle streaming vs non-streaming
        try:
            if request.stream:
                return StreamingResponse(
                    execute_graph_streaming(graph, ctx, request_id, model_name),
                    media_type="text/event-stream",
                )

            # Execute the graph
            result = await graph.arun(ctx)

            # Extract response
            response_text = result.get("response", "No response generated")

            # Build OpenAI-compatible response
            created = int(time.time())

            # Estimate tokens
            prompt_tokens = estimate_tokens(user_query)
            completion_tokens = estimate_tokens(response_text)

            return ChatCompletionResponse(
                id=request_id,
                created=created,
                model=model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(role="assistant", content=response_text),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        except HTTPException:
            # Re-raise OpenAI-shaped errors unchanged
            raise
        except Exception as exc:
            # Catch-all for unexpected errors
            raise _openai_error(
                f"Unexpected server error: {exc}",
                type_="server_error",
                status_code=500,
            )

    return app


# -----------------------------------------------------------------------------
# Example Demo (only runs when executed directly)
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import os
    import uvicorn

    # This is just a demo - users should create their own script
    # See examples/serve_openai_demo.py for a runnable version

    print("=" * 72)
    print("Demo Mode: Running with mock workers")
    print("=" * 72)
    print("For production use, create your own script that:")
    print("1. Defines your custom workers with real LLM integration")
    print("2. Builds your graph")
    print("3. Calls create_app(registry)")
    print("=" * 72 + "\n")

    # Create demo graph
    demo_graph = build_default_graph()

    # Create registry and register the demo graph
    demo_registry = GraphRegistry()
    demo_registry.register("demo", demo_graph, set_default=True)

    # Create the app
    demo_app = create_app(demo_registry)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    display_host = "localhost" if host in {"0.0.0.0", "::"} else host

    print("Starting demo server...")
    print(f"Endpoint: http://{display_host}:{port}/v1/chat/completions")
    print(f"Docs: http://{display_host}:{port}/docs\n")
    uvicorn.run(demo_app, host=host, port=port)
