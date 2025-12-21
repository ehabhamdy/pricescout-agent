import asyncio
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

from fastmcp import Client as FastMCPClient
from fastmcp.client.transports import StdioTransport

class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        title: str | None = None,
    ) -> None:
        self.name: str = name
        self.title: str | None = title
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc: list[str] = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        # Build the formatted output with title as a separate field
        output = f"Tool: {self.name}\n"

        # Add human-readable title if available
        if self.title:
            output += f"User-readable title: {self.title}\n"

        output += f"""Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

        return output

class Client:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.client: FastMCPClient | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = shutil.which("npx") if self.config["command"] == "npx" else self.config["command"]
        if command is None:
            raise ValueError(f"The command '{self.config['command']}' must be a valid executable and cannot be found.")

        server_params = StdioTransport(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]} if self.config.get("env") else None,
        )
        try:
            self.client = FastMCPClient(transport=server_params)
            await self.exit_stack.enter_async_context(self.client)
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Tool]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.client:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.client.list_tools()
        tools = []

        for tool in tools_response:
            tools.append(Tool(tool.name, tool.description or "", tool.inputSchema, tool.title))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.client:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.client.call_tool(tool_name, arguments)
                
                # Extract text content from the result
                output: list[str] = []
                for content in result.content:
                    if content.type == "text":
                        output.append(content.text)
                    elif content.type == "image":
                        output.append(f"[Image: {content.mimeType}]")
                
                return "\n".join(output)

            except Exception as e:
                attempt += 1
                logging.warning(f"Error executing tool: {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.client = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")
