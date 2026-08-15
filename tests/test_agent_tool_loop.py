import copy
import json
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from agent import Agent, AgentMemory


class FakeLLMClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def get_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append(
            {"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)}
        )
        return copy.deepcopy(next(self.responses))


@dataclass
class FakeCallToolResult:
    data: Any = None
    structured_content: dict[str, Any] | None = None
    content: list[Any] = field(default_factory=list)
    is_error: bool = False


class FakeFastMCPClient:
    def __init__(self, tool_names: list[str]) -> None:
        self.tool_names = tool_names
        self.executions: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[Any]:
        return [SimpleNamespace(name=name) for name in self.tool_names]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> FakeCallToolResult:
        self.executions.append((name, arguments))
        if name == "read_query":
            data = (
                "[{'id': 1, 'company_name': 'fireworks', "
                "'plan_name': 'DeepSeek V3', 'input_tokens': 0.56, "
                "'output_tokens': 1.68, 'currency': 'USD', "
                "'billing_period': 'per million tokens', 'features': '[]', "
                "'limitations': None, 'source_query': 'fixture', "
                "'created_at': '2026-08-13 00:00:00'}]"
            )
        else:
            data = "[{'affected_rows': -1}]"
        return FakeCallToolResult(data=data)


class FakeMCPClient:
    def __init__(self, name: str, tool_names: list[str]) -> None:
        self.name = name
        self.client = FakeFastMCPClient(tool_names)

    async def start(self) -> None:
        pass

    def connected_client(self) -> FakeFastMCPClient:
        return self.client


class AgentToolLoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.sqlite = FakeMCPClient(
            "sqlite",
            ["read_query", "write_query", "list_tables"],
        )
        self.firecrawl = FakeMCPClient(
            "firecrawl",
            ["scrape_websites", "extract_scraped_info"],
        )

    async def test_only_narrow_tools_are_exposed_and_results_keep_call_id(self) -> None:
        llm = FakeLLMClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "find_prices",
                                "arguments": (
                                    '{"providers":["fireworks"],'
                                    '"model":"DeepSeek V3"}'
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "DeepSeek V3 input costs $0.56 per million tokens.",
                },
            ]
        )
        agent = Agent([self.sqlite, self.firecrawl], llm, AgentMemory())
        await agent.initialize()

        answer = await agent.run("What does Fireworks charge for DeepSeek V3?")

        self.assertEqual(
            answer,
            "DeepSeek V3 input costs $0.56 per million tokens.",
        )
        exposed_names = {
            tool["function"]["name"] for tool in llm.calls[0]["tools"]
        }
        self.assertEqual(
            exposed_names,
            {"find_prices", "scrape_provider_pricing"},
        )
        self.assertTrue(
            {"read_query", "write_query", "scrape_websites"}.isdisjoint(
                exposed_names
            )
        )

        continuation = llm.calls[1]["messages"]
        self.assertEqual(continuation[-2]["tool_calls"][0]["id"], "call_123")
        self.assertEqual(continuation[-1]["role"], "tool")
        self.assertEqual(continuation[-1]["tool_call_id"], "call_123")
        result = json.loads(continuation[-1]["content"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["source"], "database")
        self.assertEqual(result["records"][0]["input_price"], 0.56)

    async def test_invalid_json_arguments_are_returned_to_model(self) -> None:
        llm = FakeLLMClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {
                                "name": "find_prices",
                                "arguments": "not-json",
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "I could not perform the lookup.",
                },
            ]
        )
        agent = Agent([self.sqlite, self.firecrawl], llm, AgentMemory())
        await agent.initialize()

        answer = await agent.run("Look up a price")

        self.assertEqual(answer, "I could not perform the lookup.")
        tool_result = json.loads(llm.calls[1]["messages"][-1]["content"])
        self.assertFalse(tool_result["ok"])
        self.assertIn("Invalid arguments", tool_result["error"])


if __name__ == "__main__":
    unittest.main()
