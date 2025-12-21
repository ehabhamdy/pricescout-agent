import asyncio
import logging

from config import Configuration
from mcp_client import Client
from llm_client import LLMClient
from agent import Agent, AgentMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def run() -> None:
    """Initialize and run the agent."""
    try:
        config = Configuration()
        server_config = config.load_config("servers_config.json")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return

    clients = [Client(name, srv_config) for name, srv_config in server_config.get("mcpServers", {}).items()]
    
    try:
        llm_client = LLMClient(config.llm_api_key, config.llm_provider)
    except ValueError as e:
        logging.error(f"LLM Client initialization failed: {e}")
        return

    memory = AgentMemory()
    
    agent = Agent(clients, llm_client, memory)
    
    try:
        await agent.initialize()

        while True:
            try:
                goal = input("Enter your goal (or 'quit'): ").strip()
                if goal.lower() in ["quit", "exit"]:
                    break
                
                if not goal:
                    continue

                result = await agent.run(goal)
                print("\n" + "="*50)
                print(result)
                print("="*50 + "\n")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Error during execution: {e}")
    finally:
        for client in clients:
            await client.cleanup()

def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()