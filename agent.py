import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from mcp_client import Client, Tool
from llm_client import LLMClient
from data_extractor import DataExtractor

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
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: List[int] = field(default_factory=list)

class AgentMemory:
    def __init__(self) -> None:
        self.facts: List[str] = []
        self.task_history: List[Task] = []
        self.context: Dict[str, Any] = {}

    def add_fact(self, fact: str) -> None:
        """Add a learned fact to memory."""
        if fact not in self.facts:
            self.facts.append(fact)

    def get_relevant_facts(self, query: str, max_facts: int = 5) -> List[str]:
        """Retrieve facts relevant to the current query."""
        # Simple keyword matching for now
        query_words = set(query.lower().split())
        relevant = []
        for fact in self.facts:
            fact_words = set(fact.lower().split())
            if query_words.intersection(fact_words):
                relevant.append(fact)
        
        return relevant[:max_facts]

class Agent:
    """Proactive agent that plans and executes tasks to achieve a goal."""

    def __init__(self, clients: List[Client], llm_client: LLMClient, memory: AgentMemory) -> None:
        self.clients: List[Client] = clients
        self.llm_client: LLMClient = llm_client
        self.memory: AgentMemory = memory
        self.available_tools: List[Tool] = []
        self.max_iterations: int = 10
        self._tool_client_map: Dict[str, Client] = {}
        self.data_extractor: Optional[DataExtractor] = None

    async def initialize(self) -> None:
        """Initialize all clients and discover tools."""
        for client in self.clients:
            await client.initialize()
            tools = await client.list_tools()
            self.available_tools.extend(tools)
            for tool in tools:
                self._tool_client_map[tool.name] = client
            
            if "sqlite" in client.name.lower():
                self.data_extractor = DataExtractor(client, self.llm_client)
                await self.data_extractor.setup_data_tables()
                
                # Register DataExtractor as a tool
                from mcp_client import Tool 
                self.available_tools.append(Tool(
                    name="extract_and_save_data",
                    description="Extract structured pricing data from text and save it to the database.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The text content to extract data from. Can be a reference to a previous task result using ${TASK_ID}."},
                            "source": {"type": "string", "description": "Source of the data (e.g., URL or query context)."}
                        },
                        "required": ["text"]
                    }
                ))

        logging.info(f"Agent initialized with {len(self.available_tools)} tools.")

    async def create_plan(self, goal: str) -> List[Task]:
        """Create an execution plan for the given goal."""
        system_prompt = self._build_planning_prompt(goal)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Goal: {goal}"}
        ]

        response = self.llm_client.get_response(messages)
        logging.info(f"Plan response: {response}")

        return self._parse_plan_response(response)

    def _build_planning_prompt(self, goal: str) -> str:
        """Build the system prompt for planning."""
        tools_description = "\n".join([tool.format_for_llm() for tool in self.available_tools])
        relevant_facts = self.memory.get_relevant_facts(goal)
        facts_str = "\n".join(relevant_facts) if relevant_facts else "No relevant facts found."

        return f"""You are an AI agent that creates execution plans.

Available tools:
{tools_description}

Relevant facts from memory:
{facts_str}

Instruction for Planning:
1. Batching: If multiple websites need to be scraped, ALWAYS combine them into a single `scrape_websites` call using the `websites` dictionary.
2. Data Flow: To use data from a previous task, use the syntax `${{TASK_ID}}`. For example, if Task 1 scrapes data, Task 2 can use `${{1}}` as input arguments.
3. Saving Data: ALWAYS include a task to save extracted data if new data is gathered. Use the `extract_and_save_data` tool, passing the result of the scrape task as `text`.

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

    def _parse_plan_response(self, response: str) -> List[Task]:
        """Parse the plan response from the LLM."""
        try:
            tasks = []
            
            # Strategy 1: Look for JSON code blocks
            code_blocks = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL | re.IGNORECASE)
            
            if code_blocks:
                for block in reversed(code_blocks):
                    try:
                        return self._parse_json_tasks(block)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
            
            # Strategy 2: Look for the outermost array structure
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    possible_json = response[start_idx : end_idx + 1]
                    return self._parse_json_tasks(possible_json)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logging.warning(f"Failed to parse raw extracted JSON: {e}")

            logging.error("Could not find valid JSON plan in response.")
            return []

        except Exception as e:
            logging.error(f"Failed to parse plan: {e}")
            return []



    def _substitute_variables(self, args: Any, memory: AgentMemory) -> Any:
        """Recursively substitute variables in arguments."""
        if isinstance(args, str):
            # Regex to find ${TASK_ID} pattern
            matches = re.findall(r"\$\{(\d+)\}", args)
            if not matches:
                return args
            
            new_val = args
            for task_id in matches:
                # Find task with this ID
                referred_task = next((t for t in memory.task_history if t.id == int(task_id)), None)
                if referred_task and referred_task.result:
                    # If the whole string is the variable, replace it completely (to keep types if needed)
                    if args == f"${{{task_id}}}":
                        return referred_task.result
                    # Otherwise string replacement
                    new_val = new_val.replace(f"${{{task_id}}}", str(referred_task.result))
            return new_val
            
        elif isinstance(args, dict):
            return {k: self._substitute_variables(v, memory) for k, v in args.items()}
        elif isinstance(args, list):
            return [self._substitute_variables(item, memory) for item in args]
        else:
            return args

    def _parse_json_tasks(self, json_str: str) -> List[Task]:
        """Helper to parse tasks from a JSON string."""
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

    def can_execute_task(self, task: Task, completed_tasks: Set[int]) -> bool:
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
            task.result = "No tool specified, assumed manual completion or reasoning."
            task.status = TaskStatus.COMPLETED
            return

        target_client = self._tool_client_map.get(task.tool_name)
        
        if not target_client and not task.tool_name == "extract_and_save_data":
            task.error = f"Tool {task.tool_name} not found."
            task.status = TaskStatus.FAILED
            return

        try:
            # Variable Substitution
            substituted_args = self._substitute_variables(task.tool_args or {}, self.memory) # '{"deepinfra": "| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| DeepSeek-OCR | 8k | $0.03 | $0.10 | View more |\\n| DeepSeek-V3.1-Terminus | 160k | $0.21 / $0.168 cached | $0.79 | View more |\\n| DeepSeek-V3.1 | 160k | $0.21 / $0.168 cached | $0.79 | View more |\\n| DeepSeek-V3-0324 | 160k | $0.20 / $0.106 cached | $0.88 | View more |\\n| DeepSeek-V3 | 160k | $0.32 | $0.89 | View more |\\n| DeepSeek-R1-0528 | 160k | $0.50 / $0.40 cached | $2.15 | View more |\\n| DeepSeek-R1-0528-Turbo | 32k | $1.00 | $3.00 | View more |\\n| DeepSeek-R1-Distill-Llama-70B | 128k | $0.60 | $1.20 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| DeepSeek-OCR | 8k | $0.03 | $0.10 | View more |\\n| DeepSeek-V3.1-Terminus | 160k | $0.21 / $0.168 cached | $0.79 | View more |\\n| DeepSeek-V3.1 | 160k | $0.21 / $0.168 cached | $0.79 | View more |\\n| DeepSeek-V3-0324 | 160k | $0.20 / $0.106 cached | $0.88 | View more |\\n| DeepSeek-V3 | 160k | $0.32 | $0.89 | View more |\\n| DeepSeek-R1-0528 | 160k | $0.50 / $0.40 cached | $2.15 | View more |\\n| DeepSeek-R1-0528-Turbo | 32k | $1.00 | $3.00 | View more |\\n| DeepSeek-R1-Distill-Llama-70B | 128k | $0.60 | $1.20 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Qwen3-Next-80B-A3B-Instruct | 256k | $0.09 | $1.10 | View more |\\n| Qwen3-Coder-480B-A35B-Instruct-Turbo | 256k | $0.28 | $1.20 | View more |\\n| Qwen3-Coder-480B-A35B-Instruct | 256k | $0.40 | $1.60 | View more |\\n| Qwen3-235B-A22B-Thinking-2507 | 256k | $0.23 | $2.39 | View more |\\n| Qwen3-235B-A22B-Instruct-2507 | 256k | $0.071 | $0.463 | View more |\\n| Qwen3-32B | 40k | $0.08 | $0.28 | View more |\\n| Qwen3-30B-A3B | 40k | $0.08 | $0.29 | View more |\\n| Qwen3-14B | 40k | $0.08 | $0.24 | View more |\\n| Qwen2.5-72B-Instruct | 32k | $0.12 | $0.39 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Qwen3-Next-80B-A3B-Instruct | 256k | $0.09 | $1.10 | View more |\\n| Qwen3-Coder-480B-A35B-Instruct-Turbo | 256k | $0.28 | $1.20 | View more |\\n| Qwen3-Coder-480B-A35B-Instruct | 256k | $0.40 | $1.60 | View more |\\n| Qwen3-235B-A22B-Thinking-2507 | 256k | $0.23 | $2.39 | View more |\\n| Qwen3-235B-A22B-Instruct-2507 | 256k | $0.071 | $0.463 | View more |\\n| Qwen3-32B | 40k | $0.08 | $0.28 | View more |\\n| Qwen3-30B-A3B | 40k | $0.08 | $0.29 | View more |\\n| Qwen3-14B | 40k | $0.08 | $0.24 | View more |\\n| Qwen2.5-72B-Instruct | 32k | $0.12 | $0.39 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Llama-4-Scout-17B-16E | 320k | $0.08 | $0.30 | View more |\\n| Llama-4-Maverick-17B-128E | 1024k | $0.15 | $0.60 | View more |\\n| Llama-Guard-4-12B | 160k | $0.18 | $0.18 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Llama-4-Scout-17B-16E | 320k | $0.08 | $0.30 | View more |\\n| Llama-4-Maverick-17B-128E | 1024k | $0.15 | $0.60 | View more |\\n| Llama-Guard-4-12B | 160k | $0.18 | $0.18 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Llama-3.3-70B-Instruct-Turbo | 128k | $0.10 | $0.32 | View more |\\n| Llama-3.2-11B-Vision-Instruct | 128k | $0.049 | $0.049 | View more |\\n| Llama-3.2-3B-Instruct | 128k | $0.02 | $0.02 | View more |\\n| Meta-Llama-3.1-70B-Instruct | 128k | $0.40 | $0.40 | View more |\\n| Meta-Llama-3.1-70B-Instruct-Turbo | 128k | $0.40 | $0.40 | View more |\\n| Meta-Llama-3.1-8B-Instruct | 128k | $0.03 | $0.05 | View more |\\n| Meta-Llama-3.1-8B-Instruct-Turbo | 128k | $0.02 | $0.03 | View more |\\n| Meta-Llama-3-8B-Instruct | 8k | $0.03 | $0.06 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Llama-3.3-70B-Instruct-Turbo | 128k | $0.10 | $0.32 | View more |\\n| Llama-3.2-11B-Vision-Instruct | 128k | $0.049 | $0.049 | View more |\\n| Llama-3.2-3B-Instruct | 128k | $0.02 | $0.02 | View more |\\n| Meta-Llama-3.1-70B-Instruct | 128k | $0.40 | $0.40 | View more |\\n| Meta-Llama-3.1-70B-Instruct-Turbo | 128k | $0.40 | $0.40 | View more |\\n| Meta-Llama-3.1-8B-Instruct | 128k | $0.03 | $0.05 | View more |\\n| Meta-Llama-3.1-8B-Instruct-Turbo | 128k | $0.02 | $0.03 | View more |\\n| Meta-Llama-3-8B-Instruct | 8k | $0.03 | $0.06 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| gemini-2.5-pro | 976k | $1.25 | $10.00 | View more |\\n| gemini-2.5-flash | 976k | $0.30 | $2.50 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| gemini-2.5-pro | 976k | $1.25 | $10.00 | View more |\\n| gemini-2.5-flash | 976k | $0.30 | $2.50 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| gemma-3-27b-it | 128k | $0.09 | $0.16 | View more |\\n| gemma-3-12b-it | 128k | $0.04 | $0.13 | View more |\\n| gemma-3-4b-it | 128k | $0.04 | $0.08 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| gemma-3-27b-it | 128k | $0.09 | $0.16 | View more |\\n| gemma-3-12b-it | 128k | $0.04 | $0.13 | View more |\\n| gemma-3-4b-it | 128k | $0.04 | $0.08 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Nemotron-3-Nano-30B-A3B | 256k | $0.06 | $0.24 | View more |\\n| NVIDIA-Nemotron-Nano-12B-v2-VL | 128k | $0.20 | $0.60 | View more |\\n| Llama-3.1-Nemotron-70B-Instruct | 128k | $1.20 | $1.20 | View more |\\n| Llama-3.3-Nemotron-Super-49B-v1.5 | 128k | $0.10 | $0.40 | View more |\\n| NVIDIA-Nemotron-Nano-9B-v2 | 128k | $0.04 | $0.16 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Nemotron-3-Nano-30B-A3B | 256k | $0.06 | $0.24 | View more |\\n| NVIDIA-Nemotron-Nano-12B-v2-VL | 128k | $0.20 | $0.60 | View more |\\n| Llama-3.1-Nemotron-70B-Instruct | 128k | $1.20 | $1.20 | View more |\\n| Llama-3.3-Nemotron-Super-49B-v1.5 | 128k | $0.10 | $0.40 | View more |\\n| NVIDIA-Nemotron-Nano-9B-v2 | 128k | $0.04 | $0.16 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| claude-4-opus | 195k | $16.50 | $82.50 | View more |\\n| claude-4-sonnet | 195k | $3.30 | $16.50 | View more |\\n| claude-3-7-sonnet-latest | 195k | $3.30 / $0.33 cached | $16.50 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| claude-4-opus | 195k | $16.50 | $82.50 | View more |\\n| claude-4-sonnet | 195k | $3.30 | $16.50 | View more |\\n| claude-3-7-sonnet-latest | 195k | $3.30 / $0.33 cached | $16.50 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| phi-4 | 16k | $0.07 | $0.14 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| phi-4 | 16k | $0.07 | $0.14 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Mistral-Small-3.2-24B-Instruct-2506 | 125k | $0.075 | $0.20 | View more |\\n| Mistral-Small-24B-Instruct-2501 | 32k | $0.05 | $0.08 | View more |\\n| Mistral-Nemo-Instruct-2407 | 128k | $0.02 | $0.04 | View more |\\n| Mixtral-8x7B-Instruct-v0.1 | 32k | $0.54 | $0.54 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Mistral-Small-3.2-24B-Instruct-2506 | 125k | $0.075 | $0.20 | View more |\\n| Mistral-Small-24B-Instruct-2501 | 32k | $0.05 | $0.08 | View more |\\n| Mistral-Nemo-Instruct-2407 | 128k | $0.02 | $0.04 | View more |\\n| Mixtral-8x7B-Instruct-v0.1 | 32k | $0.54 | $0.54 | View more |\\n\\n| Model | $ per minute of audio input | Actions |\\n| --- | --- | --- |\\n| Voxtral-Small-24B-2507 | $0.00300 | View more |\\n| Voxtral-Mini-3B-2507 | $0.00100 | View more |\\n\\n| Model | $ per minute of audio input | Actions |\\n| --- | --- | --- |\\n| Voxtral-Small-24B-2507 | $0.00300 | View more |\\n| Voxtral-Mini-3B-2507 | $0.00100 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Mixtral-8x7B-Instruct-v0.1 | 32k | $0.54 | $0.54 | View more |\\n| WizardLM-2-8x22B | 64k | $0.48 | $0.48 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Mixtral-8x7B-Instruct-v0.1 | 32k | $0.54 | $0.54 | View more |\\n| WizardLM-2-8x22B | 64k | $0.48 | $0.48 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Meta-Llama-3-8B-Instruct | 8k | $0.03 | $0.06 | View more |\\n| Meta-Llama-3.1-8B-Instruct | 128k | $0.03 | $0.05 | View more |\\n| gemma-3-4b-it | 128k | $0.04 | $0.08 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Meta-Llama-3-8B-Instruct | 8k | $0.03 | $0.06 | View more |\\n| Meta-Llama-3.1-8B-Instruct | 128k | $0.03 | $0.05 | View more |\\n| gemma-3-4b-it | 128k | $0.04 | $0.08 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| MythoMax-L2-13b | 4k | $0.08 | $0.08 | View more |\\n| gemma-3-27b-it | 128k | $0.09 | $0.16 | View more |\\n| gemma-3-12b-it | 128k | $0.04 | $0.13 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| MythoMax-L2-13b | 4k | $0.08 | $0.08 | View more |\\n| gemma-3-27b-it | 128k | $0.09 | $0.16 | View more |\\n| gemma-3-12b-it | 128k | $0.04 | $0.13 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Meta-Llama-3.1-70B-Instruct | 128k | $0.40 | $0.40 | View more |\\n\\n| Model | Context | $ per 1M input tokens | $ per 1M output tokens | Actions |\\n| --- | --- | --- | --- | --- |\\n| Meta-Llama-3.1-70B-Instruct | 128k | $0.40 | $0.40 | View more |\\n\\n| Model | $ per image | Actions |\\n| --- | --- | --- |\\n| FLUX-2-dev | $0.01 x (w / 1024) x (h / 1024) x (iters / 28) | View more |\\n| FLUX.1-Kontext-dev | $0.01 x (w / 1024) x (h / 1024) x (iters / 25) | View more |\\n| FLUX-1-Redux-dev | $0.012 x (w / 1024) x (h / 1024) x (iters / 25) | View more |\\n| FLUX-1-dev | $0.009 x (w / 1024) x (h / 1024) x (iters / 25) | View more |\\n| FLUX-1-schnell | $0.0005 x (w / 1024) x (h / 1024) x iters | View more |\\n| FLUX-pro | $0.05 | View more |\\n| FLUX-1.1-pro | $0.04 | View more |\\n\\n| Model | $ per image | Actions |\\n| --- | --- | --- |\\n| FLUX-2-dev | $0.01 x (w / 1024) x (h / 1024) x (iters / 28) | View more |\\n| FLUX.1-Kontext-dev | $0.01 x (w / 1024) x (h / 1024) x (iters / 25) | View more |\\n| FLUX-1-Redux-dev | $0.012 x (w / 1024) x (h / 1024) x (iters / 25) | View more |\\n| FLUX-1-dev | $0.009 x (w / 1024) x (h / 1024) x (iters / 25) | View more |\\n| FLUX-1-schnell | $0.0005 x (w / 1024) x (h / 1024) x iters | View more |\\n| FLUX-pro | $0.05 | View more |\\n| FLUX-1.1-pro | $0.04 | View more |\\n\\n| GPU | Memory | Price |\\n| --- | --- | --- |\\n| A100 | 80GB | $0.89/ GPU-hour |\\n| H100 | 80GB | $1.69/ GPU-hour |\\n| H200 | 141GB | $1.99/ GPU-hour |\\n| B200 | 180GB | $2.49/ GPU-hour |\\n\\n| Model | Context | $ per 1M input tokens |\\n| --- | --- | --- |\\n| bge-base-en-v1.5 | 512 | $0.005 |\\n| bge-en-icl | 8k | $0.01 |\\n| bge-large-en-v1.5 | 512 | $0.01 |\\n| bge-m3 | 8k | $0.01 |\\n| bge-m3-multi | 8k | $0.01 |\\n| gte-base | 512 | $0.005 |\\n| gte-large | 512 | $0.01 |\\n| e5-base-v2 | 512 | $0.005 |\\n| e5-large-v2 | 512 | $0.01 |\\n| multilingual-e5-large | 512 | $0.01 |\\n| all-MiniLM-L12-v2 | 512 | $0.005 |\\n| all-MiniLM-L6-v2 | 512 | $0.005 |\\n| all-mpnet-base-v2 | 512 | $0.005 |\\n| multi-qa-mpnet-base-dot-v1 | 512 | $0.005 |\\n| paraphrase-MiniLM-L6-v2 | 512 | $0.005 |\\n| text2vec-base-chinese | 512 | $0.005 |\\n\\n| Tier | Qualification & Invoicing Threshold | $ |\\n| --- | --- | --- |\\n| Tier 1 |  | $20 |\\n| Tier 2 | $100 paid | $100 |\\n| Tier 3 | $500 paid | $500 |\\n| Tier 4 | $2,000 paid | $2,000 |\\n| Tier 5 | $10,000 paid | $10,000 |", "fireworks": "| Base model | $ / 1M tokens |\\n| --- | --- |\\n| Less than 4B parameters | $0.10, $0.05 cached |\\n| 4B - 16B parameters | $0.20, $0.10 cached |\\n| More than 16B parameters | $0.90, $0.45 cached |\\n| MoE 0B - 56B parameters (e.g. Mixtral 8x7B) | $0.50, $0.25 cached |\\n| MoE 56.1B - 176B parameters (e.g. DBRX, Mixtral 8x22B) | $1.20, $0.60 cached |\\n| DeepSeek V3 family | $0.56 input, $0.28 cached, $1.68 output |\\n| DeepSeek R1 0528 | $1.35 input, $0.68 cached, $5.4 output |\\n| GLM-4.5,GLM-4.6 | $0.55 input, $0.28 cached, $2.19 output |\\n| Qwen3 235B Family | $0.22 input, $0.88 output |\\n| Qwen3 VL 30B A3B | $0.15 input, $0.60 output |\\n| Kimi K2 Instruct,Kimi K2 Thinking | $0.60 input, $0.30 cached, $2.50 output |\\n| Qwen3 Coder 480B | $0.45 input, $0.23 cached, $1.80 output |\\n| OpenAI gpt-oss-120b | $0.15 input, $0.07 cached, $0.60 output |\\n| OpenAI gpt-oss-20b | $0.07 input, $0.04 cached, $0.30 output |\\n| MiniMax M2 | $0.30 input, $0.15 cached, $1.20 output |\\n\\n| Model | $ / audio minute (billed per second) |\\n| --- | --- |\\n| Whisper-v3-large | $0.0015 |\\n| Whisper-v3-large-turbo | $0.0009 |\\n| Streaming ASR v1 | $0.0032 |\\n| Streaming ASR v2 | $0.0035 |\\n\\n| Image model name | $ / step | Approx $ / image |\\n| --- | --- | --- |\\n| All Non-Flux Models (SDXL, Playground, etc) | $0.00013 per step ($0.0039 per 30 step image) | $0.0002 per step ($0.006 per 30 step image) |\\n| FLUX.1 [dev] | $0.0005 per step ($0.014 per 28 step image) | N/A on serverless |\\n| FLUX.1 [schnell] | $0.00035 per step ($0.0014 per 4 step image) | N/A on serverless |\\n| FLUX.1 Kontext Pro | $0.04 per image | N/A |\\n| FLUX.1 Kontext Max | $0.08 per image | N/A |\\n\\n| Base model parameter count | $ / 1M input tokens |\\n| --- | --- |\\n| up to 150M | $0.008 |\\n| 150M - 350M | $0.016 |\\n| Qwen3 8B | $0.1 |\\n\\n| Base Model | Supervised Fine Tuning | Direct Preference Optimization |\\n| --- | --- | --- |\\n| Models up to 16B parameters | $0.50 | $1.00 |\\n| Models 16.1B - 80B | $3.00 | $6.00 |\\n| Models 80B - 300B (e.g. Qwen3-235B, gpt-oss-120B) | $6.00 | $12.00 |\\n| Models >300B (e.g. DeepSeek V3, Kimi K2) | $10.00 | $20.00 |\\n\\n| GPU Type | $ / hour (billed per second) |\\n| --- | --- |\\n| A100 80 GB GPU | $2.90 |\\n| H100 80 GB GPU | $4.00 |\\n| H200 141 GB GPU | $6.00 |\\n| B200 180 GB GPU | $9.00 |", "groq": "| AI Model | Current Speed(Tokens per Second) | Input Token Price(Per Million Tokens) | Output Token Price(Per Million Tokens) |\\n| --- | --- | --- | --- |\\n| AI ModelGPT OSS 20B 128k | Current Speed1,000 TPS | Input Token Price(Per Million Tokens)$0.075(13.3M / $1)* | Output Token Price(Per Million Tokens)$0.30(3.33M / $1)* |\\n| AI ModelGPT OSS Safeguard 20B | Current Speed1,000 TPS | Input Token Price(Per Million Tokens)$0.075(13.3M / $1)* | Output Token Price(Per Million Tokens)$0.30(3.33M / $1)* |\\n| AI ModelGPT OSS 120B 128k | Current Speed500 TPS | Input Token Price(Per Million Tokens)$0.15(6.67M / $1)* | Output Token Price(Per Million Tokens)$0.60(1.66M / $1)* |\\n| AI ModelKimi K2-0905 1T 256k | Current Speed200 TPS | Input Token Price(Per Million Tokens)$1.00(1M / $1)* | Output Token Price(Per Million Tokens)$3.00(333,333 / $1)* |\\n| AI ModelLlama 4 Scout (17Bx16E) 128k | Current Speed594 TPS | Input Token Price(Per Million Tokens)$0.11(9.09M / $1)* | Output Token Price(Per Million Tokens)$0.34(2.94M / $1)* |\\n| AI ModelLlama 4 Maverick (17Bx128E) 128k | Current Speed562 TPS | Input Token Price(Per Million Tokens)$0.20(5M / $1)* | Output Token Price(Per Million Tokens)$0.60(1.6M / $1)* |\\n| AI ModelLlama Guard 4 12B 128k | Current Speed325 TPS | Input Token Price(Per Million Tokens)$0.20(5M / $1)* | Output Token Price(Per Million Tokens)$0.20(5M / $1)* |\\n| AI ModelQwen3 32B 131k | Current Speed662 TPS | Input Token Price(Per Million Tokens)$0.29(3.44M / $1)* | Output Token Price(Per Million Tokens)$0.59(1.69M / $1)* |\\n| AI ModelLlama 3.3 70B Versatile 128k | Current Speed394 TPS | Input Token Price(Per Million Tokens)$0.59(1.69M / $1)* | Output Token Price(Per Million Tokens)$0.79(1.27M / $1)* |\\n| AI ModelLlama 3.1 8B Instant 128k | Current Speed840 TPS | Input Token Price(Per Million Tokens)$0.05(20M / $1)* | Output Token Price(Per Million Tokens)$0.08(12.5M / $1)* |\\n\\n| AI Model | Characters /s | PricePrice (Per M Characters) |\\n| --- | --- | --- |\\n| AI ModelCanopy Labs Orpheus English | Characters /s100 | Price$22.00 |\\n| AI ModelCanopy Labs Orpheus Arabic Saudi | Characters /s100 | Price$40.00 |\\n\\n| AI Model | Speed Factor | Price(Per Hour Transcribed) |\\n| --- | --- | --- |\\n| AI ModelWhisper V3 Large | Speed Factor217x | Price$0.111* |\\n| AI ModelWhisper Large v3 Turbo | Speed Factor228x | Price$0.04* |\\n\\n| Model | Uncached Input Tokens (Per M Tokens) | Cached Input Tokens (Per M Tokens) | Output Tokens (Per M Tokens) |\\n| --- | --- | --- | --- |\\n| Modelmoonshotai/kimi-k2-instruct-0905 | Uncached Input Tokens (Per M Tokens)$1.00 | Cached Input Tokens (Per M Tokens)$0.50 | Output Tokens (Per M Tokens)$3.00 |\\n| Modelopenai/gpt-oss-120b | Uncached Input Tokens (Per M Tokens)$0.15 | Cached Input Tokens (Per M Tokens)$0.075 | Output Tokens (Per M Tokens)$0.60 |\\n| Modelopenai/gpt-oss-20b | Uncached Input Tokens (Per M Tokens)$0.075 | Cached Input Tokens (Per M Tokens)$0.0375 | Output Tokens (Per M Tokens)$0.30 |\\n\\n| Tool | Price | Parameter |\\n| --- | --- | --- |\\n| ToolBasic Search | Price$5 / 1000 requests | Parameterweb_search |\\n| ToolAdvanced Search | Price$8 / 1000 requests | Parameterweb_search |\\n| ToolVisit Website | Price$1 / 1000 requests | Parametervisit_website |\\n| ToolCode Execution | Price$0.18 / hour | Parametercode_interpreter |\\n| ToolBrowser Automation | Price$0.08 / hour | Parameterbrowser_automation |\\n\\n| Tool | Price | Parameter |\\n| --- | --- | --- |\\n| ToolBrowser Search - Basic Search | Price$5 / 1000 requests | Parameterbrowser_search - browser.search |\\n| ToolBrowser Search - Visit Website | Price$1 / 1000 requests | Parameterbrowser_search - browser.open |\\n| ToolCode Execution - Python | Price$0.18 / hour | Parametercode_interpreter - python |"}'
            
            # Special handling for internal DataExtractor tool
            if task.tool_name == "extract_and_save_data" and self.data_extractor:
                logging.info(f"Executing internal tool {task.tool_name}")
                text = substituted_args.get("text", "")
                result = await self.data_extractor.extract_and_store_data(
                    substituted_args.get("source", "agent_execution"), 
                    text
                )
                task.result = "Data extracted and stored."
            
            else:
                result = await target_client.execute_tool(task.tool_name, substituted_args)
                task.result = result
            
            task.status = TaskStatus.COMPLETED
            
            self.memory.add_fact(f"Task {task.id} result: {str(result)[:200]}...")
            self.memory.task_history.append(task)

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED

    async def run(self, goal: str) -> str:
        """Run the agent loop."""
        if self.data_extractor:
            existing_data = await self.data_extractor.check_existing_prices(goal)
            if existing_data:
                logging.info(f"Found existing data for goal: {goal}")
                return existing_data

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
                
                # Logic for failed dependencies could be improved here, but skipped for brevity/preservation
                
                if self.can_execute_task(task, completed_tasks):
                    await self.execute_task(task)
                    
                    if task.status == TaskStatus.COMPLETED:
                        completed_tasks.add(task.id)
                        made_progress = True
                    elif task.status == TaskStatus.FAILED:
                        logging.error(f"Task {task.id} failed: {task.error}")
                        return f"Agent execution failed at task {task.id}: {task.error}"

            if not made_progress:
                logging.warning("No progress made in this iteration. Possible deadlock or all remaining tasks blocked.")
                break
        
        summary = "Execution completed.\n"
        all_results = []
        for task in plan:
            summary += f"Task {task.id}: {task.status.value} - {task.result or task.error}\n"
            if task.result:
                all_results.append(str(task.result))
        
        # Attempt to extract and store data from results
        if self.data_extractor and all_results:
            combined_results = "\n\n".join(all_results)
            await self.data_extractor.extract_and_store_data(goal, combined_results)

        return summary
