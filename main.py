import asyncio
import logging

from agent import Agent, AgentMemory
from config import Configuration
from llm_client import LLMClient
from mcp_client import MCPClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


async def run() -> None:
    """Initialize and run the agent."""
    try:
        config = Configuration()
        server_config = config.load_config("servers_config.json")
    except Exception as exc:
        logging.error("Failed to load configuration: %s", exc)
        return

    mcp_clients = [
        MCPClient(name, server)
        for name, server in server_config.get("mcpServers", {}).items()
    ]

    try:
        llm_client = LLMClient(config.llm_api_key, config.llm_provider)
    except ValueError as exc:
        logging.error("LLM Client initialization failed: %s", exc)
        return

    memory = AgentMemory()
    agent = Agent(mcp_clients, llm_client, memory)

    try:
        await agent.initialize()

        while True:
            try:
                goal = input("Enter your goal (or 'quit'): ").strip()
                if goal.lower() in ("quit", "exit"):
                    break

                if not goal:
                    continue

                result = await agent.run(goal)
                print("\n" + "=" * 50)
                print(result)
                print("=" * 50 + "\n")

            except KeyboardInterrupt:
                break
            except Exception as exc:
                logging.error("Error during execution: %s", exc)
    finally:
        for mcp_client in mcp_clients:
            await mcp_client.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
