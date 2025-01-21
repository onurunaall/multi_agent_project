"""
agent_logic.py
Handles agent decision logic, memory, and outcome data models.
"""

import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

import ollama  # For local LLM usage


class AgentOutcome(BaseModel):
    """
    Represents a structured decision from the Agent,
    showing which tool it wants to call, the tool input, and output.
    """
    tool_name: str
    tool_input: dict
    tool_output: Optional[str] = None

    @classmethod
    def parse_llm_output(cls, raw_llm_result: dict) -> "AgentOutcome":
        """
        Converts the LLM's JSON-like output into an AgentOutcome object.
        Raises an exception if the format is invalid.
        """
        try:
            content_data = json.loads(raw_llm_result["message"]["content"])
            return cls(
                tool_name=content_data["name"],
                tool_input=content_data["parameters"]
            )
        except Exception as exc:
            print(f"Error parsing LLM result:\n{raw_llm_result}\n")
            raise exc


def build_memory(outcomes: List[AgentOutcome], user_query: str) -> List[dict]:
    """
    Transforms past outcomes into conversation history, including a reminder
    about the user's original question. This helps the agent stay on topic.
    """
    mem: List[Dict[str, Any]] = []
    for o in outcomes:
        if o.tool_output:
            # Record the agent's decision
            mem.append({
                "role": "assistant",
                "content": json.dumps({"name": o.tool_name, "parameters": o.tool_input})
            })
            # Record what the tool returned
            mem.append({"role": "user", "content": o.tool_output})

    if mem:
        # Add a reminder of the user's main question
        mem.append({
            "role": "user",
            "content": (
                f"Reminder: The user originally asked '{user_query}'. "
                "Use what you've learned. If everything is clear, respond with the `respond_final` tool."
            )
        })
    return mem


def trigger_agent(
    system_prompt: str,
    available_tools: Dict[str, Any],
    user_q: str,
    chat_history: List[dict],
    previous_outcomes: List[AgentOutcome]
) -> AgentOutcome:
    """
    Sends the system prompt, user query, chat history, and memory to the Ollama LLM,
    then validates the output as AgentOutcome.
    """
    # Build memory from past tool usage
    memory_log = build_memory(previous_outcomes, user_q)

    # Describe the tools in a textual form for the LLM
    tools_txt = "\n".join(
        f"{idx + 1}. `{t_obj.name}`: {t_obj.description}"
        for idx, t_obj in enumerate(available_tools.values())
    )
    prompt_msg = system_prompt + "\nYou can use these tools:\n" + tools_txt

    messages = [
        {"role": "system", "content": prompt_msg},
        *chat_history,
        {"role": "user", "content": user_q},
        *memory_log
    ]

    # Request a JSON formatted response from the LLM
    llm_response = ollama.chat(model="llama3.1", messages=messages, format="json")
    return AgentOutcome.parse_llm_output(llm_response)

