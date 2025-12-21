# client - 1:1 connection
# server - Puppeteer, sqlite db
# configuration - read from json file, provide instructions on how to connect to the server(s)
# ChatSession Class - Lightweight host - needs access to LLM (groq -- llama vision model)

# CHECK - client(s) - 1:1 connection 
# server(s) - puppeteer, sqlite db 
# CHECK - configuration - read from json file, provide instructions on how to connect to the server(s)
# ChatSession - host needs access to an LLM (qroq - llama vision model)

import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

import httpx
from dotenv import load_dotenv
from fastmcp import Client as FastMCPClient
from fastmcp.client.transports import StdioTransport


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from enum import Enum
from dataclasses import dataclass, field

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class Task:
    id: int
    description: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any | None = None
    error: str | None = None
    dependencies: list[int] = field(default_factory=list)

class AgentMemory:
    def __init__(self) -> None:
        self.facts: list[str] = []
        self.task_history: list[Task] = []
        self.context: dict[str, Any] = {}

    def add_fact(self, fact: str) -> None:
        """Add a learned fact to memory."""
        if fact not in self.facts:
            self.facts.append(fact)

    def get_relevant_facts(self, query: str, max_facts: int = 5) -> list[str]:
        """Retrieve facts relevant to the current query."""
        # Simple keyword matching for now
        query_words = set(query.lower().split())
        relevant = []
        for fact in self.facts:
            fact_words = set(fact.lower().split())
            if query_words.intersection(fact_words):
                relevant.append(fact)
        
        return relevant[:max_facts]

class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("LLM_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key

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
            raise ValueError("The command must be a valid string and cannot be None.")


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

    async def list_tools(self) -> list[Any]:
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

class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        url_groq = "https://api.groq.com/openai/v1/chat/completions"
        url_vocareum = "https://claude.vocareum.com/v1/chat/completions"
        url_openai = "https://api.openai.com/v1/chat/completions"

        headers_groq = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        headers_vocareum = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        } 
        headers_openai = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload_groq = {
            "messages": messages,
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }

        payload_vocareum = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 1024,
            "messages": messages
        }

        payload_openai = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": messages
        }

        try:
            with httpx.Client() as client:
                response = client.post(url_openai, headers=headers_openai, json=payload_openai)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return f"I encountered an error: {error_message}. Please try again or rephrase your request."

class Agent:
    """Proactive agent that plans and executes tasks to achieve a goal."""

    def __init__(self, clients: list[Client], llm_client: LLMClient, memory: AgentMemory) -> None:
        self.clients: list[Client] = clients
        self.llm_client: LLMClient = llm_client
        self.memory: AgentMemory = memory
        self.available_tools: list[Tool] = []
        self.max_iterations: int = 10

    async def initialize(self) -> None:
        """Initialize all clients and discover tools."""
        for client in self.clients:
            await client.initialize()
            tools = await client.list_tools()
            self.available_tools.extend(tools)
        logging.info(f"Agent initialized with {len(self.available_tools)} tools.")

    async def create_plan(self, goal: str) -> list[Task]:
        """Create an execution plan for the given goal."""
        tools_description = "\n".join([tool.format_for_llm() for tool in self.available_tools])
        relevant_facts = self.memory.get_relevant_facts(goal)
        facts_str = "\n".join(relevant_facts) if relevant_facts else "No relevant facts found."

        system_prompt = f"""You are an AI agent that creates execution plans.

Available tools:
{tools_description}

Relevant facts from memory:
{facts_str}

Create a detailed plan to achieve the user's goal.
Return ONLY a JSON array of tasks with the following structure:
[
  {{
    "id": 1,
    "description": "task description",
    "tool_name": "exact_tool_name",
    "tool_args": {{"arg_name": "value"}},
    "dependencies": []
  }}
]
Verify tool arguments against tool descriptions. dependencies is a list of task IDs that must complete before this task.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Goal: {goal}"}
        ]

        response = self.llm_client.get_response(messages)
        logging.info(f"Plan response: {response}")

        try:
            import re
            tasks = []
            
            # Helper to parse tasks from a string
            def parse_tasks(json_str: str) -> list[Task]:
                data = json.loads(json_str)
                parsed_tasks = []
                for item in data:
                    parsed_tasks.append(Task(
                        id=item["id"],
                        description=item["description"],
                        tool_name=item.get("tool_name"),
                        tool_args=item.get("tool_args"),
                        dependencies=item.get("dependencies", [])
                    ))
                return parsed_tasks

            # Strategy 1: Look for JSON code blocks
            # Use non-greedy match .*? to capture individual blocks
            code_blocks = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL | re.IGNORECASE)
            
            if code_blocks:
                # Try the last block first (often the 'refactored' or final version)
                for block in reversed(code_blocks):
                    try:
                        tasks = parse_tasks(block)
                        if tasks: return tasks
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
            
            # Strategy 2: Look for the outermost array structure if no code blocks found or they failed
            # Find first '[' and last ']'
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    possible_json = response[start_idx : end_idx + 1]
                    tasks = parse_tasks(possible_json)
                    if tasks: return tasks
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logging.warning(f"Failed to parse raw extracted JSON: {e}")

            # If we're here, we failed to extract/parse
            logging.error("Could not find valid JSON plan in response.")
            return []

        except Exception as e:
            logging.error(f"Failed to parse plan: {e}")
            return []

    def can_execute_task(self, task: Task, completed_tasks: set[int]) -> bool:
        """Check if a task's dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in completed_tasks:
                return False
        return True

    async def execute_task(self, task: Task) -> None:
        """Execute a single task."""
        logging.info(f"Executing task {task.id}: {task.description}")
        task.status = TaskStatus.IN_PROGRESS

        if not task.tool_name:
            # Maybe it's a reasoning task or manual task?
            task.result = "No tool specified, assumed manual completion or reasoning."
            task.status = TaskStatus.COMPLETED
            return

        # Find client for tool
        target_client = None
        for client in self.clients:
            tools = await client.list_tools()
            if any(t.name == task.tool_name for t in tools):
                target_client = client
                break
        
        if not target_client:
            task.error = f"Tool {task.tool_name} not found."
            task.status = TaskStatus.FAILED
            return

        try:
            result = await target_client.execute_tool(task.tool_name, task.tool_args or {})
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            # Extract facts?
            self.memory.add_fact(f"Task {task.id} result: {str(result)[:200]}...")

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED

    async def run(self, goal: str) -> str:
        """Run the agent loop."""
        # await self.initialize()
        
        plan = await self.create_plan(goal)
        if not plan:
            return "Failed to create a plan."

        logging.info(f"Created plan with {len(plan)} tasks.")
        
        completed_tasks = set()
        iterations = 0

        while len(completed_tasks) < len(plan) and iterations < self.max_iterations:
            iterations += 1
            made_progress = False
            
            pending_tasks = [t for t in plan if t.id not in completed_tasks]
            if not pending_tasks:
                break

            for task in plan:
                if task.id in completed_tasks:
                    continue
                
                if task.status == TaskStatus.FAILED:
                    # Depending on policy, we might stop or continue. For now, let's stop if dependency failed.
                    pass 

                if self.can_execute_task(task, completed_tasks):
                    await self.execute_task(task)
                    
                    if task.status == TaskStatus.COMPLETED:
                        completed_tasks.add(task.id)
                        made_progress = True
                    elif task.status == TaskStatus.FAILED:
                        logging.error(f"Task {task.id} failed: {task.error}")
                        # We might need to abort or replan. For simplicity, abort.
                        return f"Agent execution failed at task {task.id}: {task.error}"

            if not made_progress:
                logging.warning("No progress made in this iteration. Possible deadlock or all remaining tasks blocked.")
                break
        
        summary = "Execution completed.\n"
        for task in plan:
            summary += f"Task {task.id}: {task.status.value} - {task.result or task.error}\n"
        
        return summary

async def run() -> None:
    """Initialize and run the agent."""
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    clients = [Client(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    llm_client = LLMClient(config.llm_api_key)
    memory = AgentMemory()
    
    agent = Agent(clients, llm_client, memory)
    await agent.initialize()

    try:
        while True:
            try:
                goal = input("Enter your goal (or 'quit'): ").strip()
                if goal.lower() in ["quit", "exit"]:
                    break
                
                result = await agent.run(goal)
                print("\n" + "="*50)
                print(result)
                print("="*50 + "\n")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Error during execution: {e}")
    finally:
        # Cleanup
        for client in clients:
            await client.cleanup()

def main() -> None:
    asyncio.run(run())

if __name__ == "__main__":
    main()