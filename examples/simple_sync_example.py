"""
Simple Synchronous Example

Shows how to use thinkagain with sync workers in non-async code.
This is useful for simple scripts or gradual migration to async.

Run with: python examples/simple_sync_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import Context, Worker


# ============================================================================
# Simple Sync Workers
# ============================================================================

class Step1(Worker):
    def __call__(self, ctx: Context) -> Context:
        ctx.result = "Step 1 complete"
        ctx.log(f"[{self.name}] Processing...")
        return ctx


class Step2(Worker):
    def __call__(self, ctx: Context) -> Context:
        ctx.result = f"{ctx.result} → Step 2 complete"
        ctx.log(f"[{self.name}] Processing...")
        return ctx


class Step3(Worker):
    def __call__(self, ctx: Context) -> Context:
        ctx.result = f"{ctx.result} → Step 3 complete"
        ctx.log(f"[{self.name}] Processing...")
        return ctx


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Simple Synchronous Pipeline Example")
    print("=" * 70)

    # Build pipeline with >> operator
    pipeline = Step1() >> Step2() >> Step3()

    print(f"\nPipeline created with {len(pipeline.nodes)} steps")
    print("Executing synchronously (no await needed)...\n")

    # Execute synchronously - no asyncio.run() needed!
    # The framework handles it internally
    ctx = Context(data="input")
    result = pipeline(ctx)

    print(f"\nResult: {result.result}")
    print("\n✅ Sync execution works seamlessly!")
    print("=" * 70)


if __name__ == "__main__":
    main()
