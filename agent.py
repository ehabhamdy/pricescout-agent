import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from llm_client import LLMClient
from mcp_client import MCPClient
from price_tools import PriceTools


class AgentMemory:
    """Small conversational memory containing only completed chat turns."""

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        self.messages: List[Dict[str, Any]] = []

    def add_exchange(self, user_message: str, assistant_message: str) -> None:
        self.messages.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        self.messages = self.messages[-(self.max_turns * 2):]

    def get_messages(self) -> List[Dict[str, Any]]:
        return [message.copy() for message in self.messages]


class Agent:
    """Run an LLM tool loop over tools discovered from MCP servers."""

    def __init__(
        self,
        mcp_clients: List[MCPClient],
        llm_client: LLMClient,
        memory: AgentMemory,
    ) -> None:
        self.mcp_clients = mcp_clients
        self.llm_client = llm_client
        self.memory = memory
        self.max_iterations = 10
        self.price_tools: PriceTools | None = None

    async def initialize(self) -> None:
        """Connect to every MCP server and discover its tools."""
        sqlite_client = None
        firecrawl_client = None
        sqlite_tools: set[str] = set()
        firecrawl_tools: set[str] = set()

        for mcp_client in self.mcp_clients:
            await mcp_client.start()
            client = mcp_client.connected_client()
            tool_names = {tool.name for tool in await client.list_tools()}
            if "sqlite" in mcp_client.name.lower():
                sqlite_client = client
                sqlite_tools = tool_names
            if "firecrawl" in mcp_client.name.lower():
                firecrawl_client = client
                firecrawl_tools = tool_names

        sqlite_requirements = {"read_query", "write_query"}
        if not sqlite_client or not sqlite_requirements.issubset(sqlite_tools):
            raise RuntimeError("SQLite MCP must provide read_query and write_query")
        if not firecrawl_client or "scrape_websites" not in firecrawl_tools:
            raise RuntimeError("Firecrawl MCP must provide scrape_websites")

        self.price_tools = PriceTools(
            sqlite_client,
            firecrawl_client,
            self._provider_catalog(),
        )
        await self.price_tools.initialize()
        logging.info("Agent initialized with 3 narrow pricing tools.")

    def _provider_catalog(self) -> Dict[str, str]:
        path = Path(__file__).with_name("providers.json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            providers = {}
            for name, url in data.items():
                if not isinstance(name, str) or not isinstance(url, str):
                    continue
                parsed_url = urlparse(url)
                if parsed_url.scheme in {"http", "https"} and parsed_url.netloc:
                    providers[name] = url
                else:
                    logging.warning("Ignoring invalid provider URL for %s", name)
            return providers
        except (OSError, json.JSONDecodeError, TypeError):
            logging.warning("Could not load provider catalog from %s", path)
            return {}

    def _system_prompt(self) -> str:
        return """You are PriceScout, a pricing research assistant.

Use the provided narrow pricing tools whenever data is needed. Do not invent
tool results, providers, models, URLs, or prices.

For pricing requests:
1. Call find_prices first for every requested provider and model.
2. If missing_providers is empty, answer from those database records.
3. Otherwise call scrape_provider_pricing for only the missing providers.
4. Extract only prices explicitly present in the scrape content and pass them to
   save_prices. Never infer or manufacture a missing price.
5. Call find_prices again to verify saved records before answering.
6. Treat ok=false as a tool failure and explain any unresolved missing data.

After using the tools, answer the user directly. State whether the answer came
from cached database rows or a fresh scrape. Do not return an execution plan or
an internal task summary.
"""

    def _function_tools(self) -> List[Dict[str, Any]]:
        if not self.price_tools:
            raise RuntimeError("Agent is not initialized")
        return self.price_tools.model_tools()

    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        function = tool_call.get("function", {})
        tool_name = function.get("name", "")
        raw_arguments = function.get("arguments", "{}")

        try:
            arguments = json.loads(raw_arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return json.dumps(
                {"ok": False, "error": f"Invalid arguments for {tool_name}: {exc}"}
            )

        logging.info("Model requested tool %s", tool_name)

        try:
            if not self.price_tools:
                raise RuntimeError("Agent is not initialized")
            return await self.price_tools.execute(tool_name, arguments)
        except Exception as exc:
            logging.warning("Tool %s failed: %s", tool_name, exc)
            return json.dumps({"ok": False, "error": str(exc)})

    async def run(self, goal: str) -> str:
        """Run model -> tool -> result iterations until the model answers."""
        if not self.price_tools:
            raise RuntimeError("Agent is not initialized")
        self.price_tools.reset_turn()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            *self.memory.get_messages(),
            {"role": "user", "content": goal},
        ]
        for iteration in range(1, self.max_iterations + 1):
            logging.info("Starting model iteration %s", iteration)
            assistant_message = self.llm_client.get_completion(
                messages,
                self._function_tools(),
            )
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []

            # If no tool calls required, reply with the final answer
            if not tool_calls:
                final_answer = (assistant_message.get("content") or "").strip()
                if not final_answer:
                    final_answer = "I could not produce a final answer."
                self.memory.add_exchange(goal, final_answer)
                return final_answer

            # Otherwise execute the tools and include the results of the tool calls in the messages to be send to the model
            for tool_call in tool_calls:
                tool_result = await self._execute_tool_call(tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": tool_result,
                    }
                )

        return (
            f"I stopped after {self.max_iterations} model iterations without "
            "receiving a final answer."
        )
