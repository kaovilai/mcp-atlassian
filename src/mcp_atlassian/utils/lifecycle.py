"""Lifecycle management utilities for graceful shutdown and signal handling."""

import logging
import os
import signal
import sys
from typing import Any

logger = logging.getLogger("mcp-atlassian.utils.lifecycle")

_shutting_down = False


def _handle_termination(signum: int, frame: Any) -> None:
    """Start shutdown on the first signal; force exit on a repeat."""
    global _shutting_down
    if _shutting_down:
        os._exit(128 + signum)
    _shutting_down = True
    raise KeyboardInterrupt


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown.

    SIGINT keeps Python's default handler, so ``asyncio.run()`` cancels the main
    task on the first Ctrl+C. SIGTERM raises KeyboardInterrupt in the main
    thread, so the server unwinds and ``main()`` exits; a second SIGTERM during
    that shutdown exits immediately.
    SIGPIPE (Unix/Linux only) is ignored so writes to a closed pipe raise
    BrokenPipeError instead of killing the process.

    Platform Behavior:
        - Unix/Linux: SIGPIPE ignored to prevent process termination on client disconnect
        - Windows: SIGPIPE not available (socket errors returned directly instead)

    This is particularly important for:
        - MCP stdio transport (client disconnect detection)
        - Docker containers running with the -i flag
        - Long-running server processes with unreliable clients

    Note:
        The handlers must not swallow termination signals or take locks: a
        handler that only records the signal leaves the process running after
        the client is gone, and one that blocks on a lock can deadlock the
        event loop when signals arrive back to back.
    """
    # Register signal handlers
    signal.signal(signal.SIGTERM, _handle_termination)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    # Handle SIGPIPE which occurs when parent process closes the pipe
    # SIGPIPE is not available on Windows, so we check for it first
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
        logger.debug("SIGPIPE ignored")
    else:
        # SIGPIPE may not be available on all platforms (e.g., Windows)
        logger.debug("SIGPIPE not available on this platform")


def ensure_clean_exit() -> None:
    """Ensure all output streams are flushed before exit.

    This is important for containerized environments where output might be
    buffered and could be lost if not properly flushed before exit.

    Handles cases where streams may already be closed by the parent process,
    particularly on Windows or when run as a child process.
    """
    logger.info("Server stopped, flushing output streams...")

    # Safely flush stdout
    try:
        if hasattr(sys.stdout, "closed") and not sys.stdout.closed:
            sys.stdout.flush()
    except (ValueError, OSError, AttributeError) as e:
        # Stream might be closed or redirected
        logger.debug(f"Could not flush stdout: {e}")

    # Safely flush stderr
    try:
        if hasattr(sys.stderr, "closed") and not sys.stderr.closed:
            sys.stderr.flush()
    except (ValueError, OSError, AttributeError) as e:
        # Stream might be closed or redirected
        logger.debug(f"Could not flush stderr: {e}")

    logger.debug("Output streams flushed, exiting gracefully")
