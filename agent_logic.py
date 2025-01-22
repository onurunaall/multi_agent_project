"""
agent_logic.py
Handles agent decision logic, memory, and outcome data models.
Implements:
 - Adaptive Prompting (dynamic prompt changes for ambiguous queries or repeated failures)
 - Autonomous Feedback Loops (self-evaluation of responses and iterative improvement)
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


# ------------------------- Adaptive Prompting Helpers -------------------------
def is_query_ambiguous(user_q: str) -> bool:
    """
    Simple check to see if a query might be ambiguous.
    You can replace this with advanced NLP if desired.
    """
    return len(user_q.split()) < 3


def adapt_system_prompt(base_prompt: str, user_q: str, attempt_count: int) -> str:
    """
    Dynamically modifies the system prompt if the query is ambiguous
    or if we've retried multiple times (attempt_count > 0).
    """
    adapted_prompt = base_prompt

    # If the query is short/ambiguous, ask the agent to request clarification.
    if is_query_ambiguous(user_q):
        adapted_prompt += (
            "\nThe user's request seems unclear. Please prompt the user for more details."
        )

    # If we've already tried at least once, push more explicit instructions.
    if attempt_count > 0:
        adapted_prompt += (
            "\nIt seems you've made an attempt before. If still uncertain, explicitly ask the user for clarification."
        )

    return adapted_prompt


# ------------------------- Autonomous Feedback Loops Helper -------------------------
def evaluate_agent_outcome(agent_outcome: AgentOutcome) -> float:
    """
    Evaluate the agent's chosen tool/output. This scoring is simplistic:
    - If the agent is uncertain or the tool name is empty, score is low.
    - Real implementations can do more advanced checks.
    """
    score = 1.0  # start with a perfect score

    # If agent picked no tool or gave no name, that's a bad sign
    if not agent_outcome.tool_name:
        score -= 0.5

    # If agent outcome suggests confusion
    # e.g., if the tool_input is suspiciously empty or a self-indication of uncertainty
    if "not sure" in str(agent_outcome.tool_input).lower():
        score -= 0.5

    return score


# ------------------------- Memory Builder -------------------------
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


# ------------------------- The Core Agent Trigger -------------------------
def trigger_agent(
    system_prompt: str,
    available_tools: Dict[str, Any],
    user_q: str,
    chat_history: List[dict],
    previous_outcomes: List[AgentOutcome],
    attempt_count: int = 0
) -> AgentOutcome:
    """
    Sends the system prompt, user query, chat history, and memory to the Ollama LLM,
    then validates the output as AgentOutcome.
    Incorporates:
      - Adaptive Prompting (adapt_system_prompt)
      - Called externally with attempt_count for repeated tries
    """
    # Adapt the system prompt based on the query and attempt count
    adapted_prompt = adapt_system_prompt(system_prompt, user_q, attempt_count)

    # Build memory from past tool usage
    memory_log = build_memory(previous_outcomes, user_q)

    # Describe the tools in a textual form for the LLM
    tools_txt = "\n".join(
        f"{idx + 1}. `{t_obj.name}`: {t_obj.description}"
        for idx, t_obj in enumerate(available_tools.values())
    )
    final_prompt = adapted_prompt + "\nYou can use these tools:\n" + tools_txt

    # Build messages
    messages = [
        {"role": "system", "content": final_prompt},
        *chat_history,
        {"role": "user", "content": user_q},
        *memory_log
    ]

    # LLM call
    llm_response = ollama.chat(model="llama3.1", messages=messages, format="json")

    # Parse output
    outcome = AgentOutcome.parse_llm_output(llm_response)

    # Autonomous feedback loop: Evaluate the outcome
    quality_score = evaluate_agent_outcome(outcome)
    # If the quality is too low, we might raise an exception or mark it for re-try
    # For demonstration, we just print a note
    if quality_score < 0.9:
        print(f"[DEBUG] Low confidence: {quality_score:.2f}, might consider re-trying or refining further.")

    return outcome
