import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from mcp_client import Client, Tool
from llm_client import LLMClient

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

    async def initialize(self) -> None:
        """Initialize all clients and discover tools."""
        for client in self.clients:
            await client.initialize()
            tools = await client.list_tools()
            self.available_tools.extend(tools)
            for tool in tools:
                self._tool_client_map[tool.name] = client
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
        
        if not target_client:
            task.error = f"Tool {task.tool_name} not found."
            task.status = TaskStatus.FAILED
            return

        try:
            result = await target_client.execute_tool(task.tool_name, task.tool_args or {})
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            self.memory.add_fact(f"Task {task.id} result: {str(result)[:200]}...")

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED

    async def run(self, goal: str) -> str:
        """Run the agent loop."""
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
        for task in plan:
            summary += f"Task {task.id}: {task.status.value} - {task.result or task.error}\n"
        
        return summary
