import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

from fastmcp import Client as FastMCPClient
from fastmcp.client.transports import StdioTransport


class MCPClient:
    """Own one stdio MCP server process and its FastMCP client lifetime."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self.client: FastMCPClient | None = None
        self._exit_stack = AsyncExitStack()

    async def start(self) -> None:
        """Start the configured MCP server and connect a FastMCP client."""
        configured_command = self.config["command"]
        command = (
            shutil.which("npx")
            if configured_command == "npx"
            else configured_command
        )
        if command is None:
            raise ValueError(
                f"The command '{configured_command}' could not be found."
            )

        transport = StdioTransport(
            command=command,
            args=self.config["args"],
            env=(
                {**os.environ, **self.config["env"]}
                if self.config.get("env")
                else None
            ),
        )
        try:
            client = FastMCPClient(transport=transport)
            await self._exit_stack.enter_async_context(client)
            self.client = client
        except Exception:
            await self.close()
            raise

    def connected_client(self) -> FastMCPClient:
        """Return the active FastMCP client or fail before any MCP operation."""
        if self.client is None:
            raise RuntimeError(f"MCP server {self.name} is not connected")
        return self.client

    async def close(self) -> None:
        """Close the MCP session and its stdio server process."""
        await self._exit_stack.aclose()
        self.client = None
