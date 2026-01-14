"""CLI tool to start a service server."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import sys

from .server import serve


def main():
    """Entry point for service server CLI.

    Usage:
        python -m thinkagain.serve my_module:my_function --port 8000
    """
    parser = argparse.ArgumentParser(
        description="Start a ThinkAgain service server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with a @service decorated class
  python -m thinkagain.serve examples.llm:LLMServer --port 8000

  # The target must be a class decorated with @service:

  @service(gpus=1, backend="grpc")
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

    # Get the @service decorated object
    try:
        obj = getattr(module, obj_name)
    except AttributeError:
        print(
            f"Error: '{obj_name}' not found in module '{module_name}'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check if it's @service decorated
    if not hasattr(obj, "_service_config"):
        print(
            f"Error: {obj_name} must be decorated with @service",
            file=sys.stderr,
        )
        print("Example:", file=sys.stderr)
        print("  @service(backend='grpc')", file=sys.stderr)
        print("  class MyService:", file=sys.stderr)
        print("      async def __call__(self, ...): ...", file=sys.stderr)
        sys.exit(1)

    # Must be a class
    if not inspect.isclass(obj):
        print(
            f"Error: {obj_name} must be a class decorated with @service",
            file=sys.stderr,
        )
        sys.exit(1)

    # Instantiate the service class
    try:
        instance = obj()
    except Exception as e:
        print(
            f"Error: Failed to instantiate {obj_name}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify callable
    if not callable(instance):
        print(
            f"Error: {obj_name} instance must be callable (needs __call__ method)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Serving @service decorated: {obj_name}")
    print(f"Backend: {obj._service_config.backend}")
    if obj._service_config.gpus:
        print(f"GPUs: {obj._service_config.gpus}")

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
