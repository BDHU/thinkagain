"""
Shared worker implementations for RAG examples.

This module contains reusable worker implementations to avoid duplication
across example files.
"""

from src.core import Worker, Context


class VectorDBWorker(Worker):
    """Simulates vector database retrieval."""

    def __init__(self, name: str = None, collection: str = "documents"):
        super().__init__(name or "vector_db")
        self.collection = collection
        self.documents = {
            "ml_basics": "Machine learning is a subset of AI that enables systems to learn from data.",
            "deep_learning": "Deep learning uses neural networks with multiple layers to learn representations.",
            "transformers": "Transformers are a neural network architecture based on self-attention mechanisms.",
            "rag": "RAG combines retrieval systems with generative models for better responses.",
            "embeddings": "Embeddings are vector representations of text that capture semantic meaning.",
        }

    def __call__(self, ctx: Context) -> Context:
        top_k = ctx.get("top_k", 3)
        results = self.search(ctx.query, top_k)
        ctx.documents = results
        ctx.log(f"[{self.name}] Retrieved {len(results)} documents from '{self.collection}'")
        return ctx

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Simple keyword-based search simulation."""
        query_lower = query.lower()
        scored_docs = []

        for doc_id, content in self.documents.items():
            score = sum(1 for word in query_lower.split() if word in content.lower())
            if score > 0:
                scored_docs.append((content, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]


class WebSearchWorker(Worker):
    """Fallback web search when vector DB has insufficient results."""

    def __init__(self, name: str = None):
        super().__init__(name or "web_search")

    def __call__(self, ctx: Context) -> Context:
        web_results = [
            f"Web result for '{ctx.query}': Additional information from the internet.",
            f"Web source 2: {ctx.query} is an important topic in AI."
        ]
        existing = ctx.get("documents", [])
        ctx.documents = existing + web_results
        ctx.log(f"[{self.name}] Added {len(web_results)} web search results")
        return ctx


class RerankerWorker(Worker):
    """Reranks documents for better relevance."""

    def __init__(self, name: str = None):
        super().__init__(name or "reranker")

    def __call__(self, ctx: Context) -> Context:
        if not ctx.documents:
            ctx.log(f"[{self.name}] No documents to rerank")
            return ctx

        top_n = ctx.get("rerank_top_n", 2)
        reranked = self.rerank(ctx.documents, top_n)
        ctx.documents = reranked
        ctx.log(f"[{self.name}] Reranked to top {len(reranked)} documents")
        return ctx

    def rerank(self, documents: list[str], top_n: int) -> list[str]:
        """Simple reranking by document length."""
        scored = [(doc, len(doc)) for doc in documents]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_n]]


class QueryRefinerWorker(Worker):
    """Refines the query to improve retrieval."""

    def __init__(self, name: str = None):
        super().__init__(name or "query_refiner")

    def __call__(self, ctx: Context) -> Context:
        original_query = ctx.query
        ctx.query = f"{original_query} detailed explanation"
        ctx.log(f"[{self.name}] Refined query from '{original_query}' to '{ctx.query}'")
        return ctx


class GeneratorWorker(Worker):
    """Generates final answer using LLM."""

    def __init__(self, name: str = None, model: str = "simulated-llm"):
        super().__init__(name or "generator")
        self.model = model

    def __call__(self, ctx: Context) -> Context:
        context_docs = ctx.get("documents", [])
        prompt = self.build_prompt(ctx.query, context_docs)
        ctx.prompt = prompt
        ctx.answer = self.generate(prompt)
        ctx.log(f"[{self.name}] Generated answer using {self.model}")
        return ctx

    def build_prompt(self, query: str, documents: list[str]) -> str:
        """Build prompt for the LLM."""
        context = "\n\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))
        return f"""Based on the following documents, answer the question.

Context:
{context}

Question: {query}

Answer:"""

    def generate(self, prompt: str) -> str:
        """Simulate LLM response."""
        return "Based on the provided context, here's the answer. (Simulated response)"
