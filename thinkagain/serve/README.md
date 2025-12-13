# ThinkAgain OpenAI-Compatible Server

This package exposes ThinkAgain graphs behind an OpenAI-style `/v1/chat/completions` API using FastAPI. It is intended to be imported and configured from your own script.

## Install

```bash
pip install -e ".[serve]"
```

This pulls in `fastapi`, `uvicorn`, and friends.

## Minimal example

`my_server.py`:

```python
from thinkagain import Context, Worker, Graph, END
from thinkagain.serve.openai.serve_completion import create_app, GraphRegistry


class MyWorker(Worker):
    async def arun(self, ctx: Context) -> Context:
        ctx.response = f"Response to: {ctx.user_query}"
        return ctx


graph = Graph(name="my_graph")
graph.add_node("worker", MyWorker())
graph.add_edge("worker", END)

registry = GraphRegistry()
registry.register("my-model", graph, set_default=True)

app = create_app(registry)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run:

```bash
python my_server.py
# or
uvicorn my_server:app --reload
```

## Calling the API

With `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

With the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

resp = client.chat.completions.create(
    model="my-model",  # must match the name you registered
    messages=[{"role": "user", "content": "What is ThinkAgain?"}],
)
print(resp.choices[0].message.content)
```

## Endpoints

- `POST /v1/chat/completions` – chat completions (streaming and non‑streaming)
- `GET /v1/models` – list registered graphs (models)
- `GET /health` and `GET /` – health + basic info
- `GET /docs` and `GET /redoc` – interactive API docs

## Context fields

For each request, the server populates your `Context`, including:

- `ctx.user_query` – last user message
- `ctx.messages` – full conversation history
- `ctx.temperature`, `ctx.max_tokens`, etc. – request params

Use these in your workers as needed.

## Production

Use an ASGI server in front of the app:

```bash
gunicorn my_server:app -w 4 -k uvicorn.workers.UvicornWorker
```

or:

```bash
uvicorn my_server:app --host 0.0.0.0 --port 8000 --workers 4
```

## Demo server

Run the built-in demo server:

```bash
python -m thinkagain.serve.openai
```

This starts a demo server with a mock graph on `http://localhost:8000`.

For a more complete example, see `examples/serve_openai_demo.py` and the library implementation in `thinkagain/serve/openai/serve_completion.py`.
