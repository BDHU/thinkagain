"""CLI tool to start a replica server."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import sys

from .server import serve


def main():
    """Entry point for replica server CLI.

    Usage:
        python -m thinkagain.serve my_module:my_function --port 8000
    """
    parser = argparse.ArgumentParser(
        description="Start a ThinkAgain replica server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with a @replicate decorated function
  python -m thinkagain.serve examples.text_processing:reverse_text --port 8000

  # Start server with a @replicate decorated class
  python -m thinkagain.serve examples.llm:LLMServer --port 8000

  # The target must be decorated with @replicate:

  # Function example:
  @replicate(backend="grpc")
  async def reverse_text(text: str) -> str:
      return text[::-1]

  # Class example:
  @replicate(gpus=1, backend="grpc")
  class LLMServer:
      def __init__(self):
          self.model = load_model()

      async def __call__(self, prompt: str) -> str:
          return self.model.generate(prompt)
        """,
    )

    parser.add_argument(
        "target",
        help="Module and function/class name in format 'module:name' or 'module.submodule:name'",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )

    args = parser.parse_args()

    # Parse target
    if ":" not in args.target:
        print(f"Error: Invalid target format '{args.target}'", file=sys.stderr)
        print("Expected format: 'module:name'", file=sys.stderr)
        sys.exit(1)

    module_name, obj_name = args.target.split(":", 1)

    # Import module
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error: Failed to import module '{module_name}': {e}", file=sys.stderr)
        sys.exit(1)

    # Get the @replicate decorated object
    try:
        obj = getattr(module, obj_name)
    except AttributeError:
        print(
            f"Error: '{obj_name}' not found in module '{module_name}'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check if it's @replicate decorated
    if not hasattr(obj, "_distribution_config"):
        print(
            f"Error: {obj_name} must be decorated with @replicate",
            file=sys.stderr,
        )
        print("Example:", file=sys.stderr)
        print("  @replicate(backend='grpc')", file=sys.stderr)
        print("  async def my_function(...): ...", file=sys.stderr)
        sys.exit(1)

    # Get the underlying function/class
    underlying = getattr(obj, "_fn", obj)

    # If it's a class, instantiate it
    # If it's a function, wrap it in a callable instance
    try:
        if inspect.isclass(underlying):
            # It's a class - instantiate it
            instance = underlying()
        else:
            # It's a function - create a wrapper instance
            class FunctionWrapper:
                def __init__(self, fn):
                    self.fn = fn

                async def __call__(self, *args, **kwargs):
                    return await self.fn(*args, **kwargs)

            instance = FunctionWrapper(underlying)
    except Exception as e:
        print(
            f"Error: Failed to instantiate {obj_name}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify callable
    if not callable(instance):
        print(
            f"Error: {obj_name} instance must be callable",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Serving @replicate decorated: {obj_name}")
    print(f"Backend: {obj._distribution_config.backend}")
    if obj._distribution_config.gpus:
        print(f"GPUs: {obj._distribution_config.gpus}")

    # Start server
    try:
        asyncio.run(serve(instance, port=args.port))
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error: Server failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
