"""
workflow_builder.py
Defines the workflow using LangGraph, connecting Agents and Tools.
"""

from typing import TypedDict, List, Dict, Any
from langchain.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod
from agent_logic import AgentOutcome, trigger_agent
from tools import duckduckgo_search, respond_final, wiki_lookup


class AgentState(TypedDict):
    """
    State structure for the workflow. 
    Tracks the user query, chat history, agent outcomes, and final output.
    """
    user_q: str
    chat_history: List[dict]
    lst_res: List[AgentOutcome]
    output: Dict[str, Any]


# ---------------------- Node Functions ----------------------
def primary_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--- Primary Agent Node ---")
    system_prompt = (
        "You are the first Agent. Decide if you need to search the web or finalize."
        "\nReturn JSON with {\"name\":\"<tool_name>\", \"parameters\":{...}}"
    )
    used_tools = {
        "browse_duckduckgo": duckduckgo_search,
        "respond_final": respond_final
    }
    result = trigger_agent(
        system_prompt=system_prompt,
        available_tools=used_tools,
        user_q=state["user_q"],
        chat_history=state["chat_history"],
        previous_outcomes=state["lst_res"]
    )
    return {"lst_res": [result]}


def second_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--- Second Agent Node ---")
    system_prompt = (
        "You are the second Agent. You can only use Wikipedia once, then finalize."
        "\nReturn JSON with {\"name\":\"<tool_name>\", \"parameters\":{...}}"
    )
    used_tools = {
        "tool_wikipedia": wiki_lookup,
        "respond_final": respond_final
    }
    # Use the previous output as a 'user_q'
    prev_output = state["output"].get("tool_output", "")
    result = trigger_agent(
        system_prompt=system_prompt,
        available_tools=used_tools,
        user_q=prev_output,
        chat_history=state["chat_history"],
        previous_outcomes=state["lst_res"]
    )
    return {"lst_res": [result]}


def tool_node(state: AgentState) -> Dict[str, Any]:
    print("--- Tool Node ---")
    last_res = state["lst_res"][-1]
    tool_map = {
        "browse_duckduckgo": duckduckgo_search,
        "respond_final": respond_final,
        "tool_wikipedia": wiki_lookup
    }
    chosen_tool = tool_map.get(last_res.tool_name)
    if not chosen_tool:
        return {}

    tool_output_str = chosen_tool(**last_res.tool_input)
    updated_res = AgentOutcome(
        tool_name=last_res.tool_name,
        tool_input=last_res.tool_input,
        tool_output=str(tool_output_str)
    )
    if updated_res.tool_name == "respond_final":
        return {"output": updated_res}
    return {"lst_res": [updated_res]}


def human_feedback_node(state: AgentState) -> None:
    """
    Dummy node to represent a 'human in the loop' step. 
    We just pass here, actual logic is in the next function.
    """
    pass


def check_human_decision(state: AgentState) -> str:
    print("--- Checking Human Decision ---")
    user_choice = input("Should we continue to the second Agent? [y/n]: ")
    return "Agent2" if user_choice.lower().startswith("y") else END


# Decide next node based on the last tool name
def determine_next_tool(state: AgentState) -> str:
    print("--- Determining Next Tool ---")
    last_outcome = state["lst_res"][-1]
    return last_outcome.tool_name if state["lst_res"] else "respond_final"


# ---------------------- Workflow Builders ----------------------
def build_single_agent_flow() -> StateGraph[AgentState]:
    """
    Simple workflow for a single Agent that can do a web search or finalize.
    """
    graph = StateGraph(AgentState)
    graph.add_node("Agent1", action=primary_agent_node)
    graph.set_entry_point("Agent1")

    graph.add_node("browse_duckduckgo", action=tool_node)
    graph.add_node("respond_final", action=tool_node)

    graph.add_conditional_edges("Agent1", determine_next_tool)
    graph.add_edge("browse_duckduckgo", "Agent1")
    graph.add_edge("respond_final", END)

    return graph.compile()


def build_multi_agent_flow() -> StateGraph[AgentState]:
    """
    More advanced flow: 
      1) Primary Agent
      2) Possibly ask human
      3) Second Agent
      4) Tools
    """
    graph = StateGraph(AgentState)

    # Agent 1
    graph.add_node("Agent1", action=primary_agent_node)
    graph.set_entry_point("Agent1")

    # Tools for Agent 1
    graph.add_node("browse_duckduckgo", action=tool_node)
    graph.add_node("respond_final", action=tool_node)

    # Next step after Agent 1
    graph.add_conditional_edges("Agent1", determine_next_tool)
    graph.add_edge("browse_duckduckgo", "Agent1")

    # Human in the loop
    graph.add_node("Human", action=human_feedback_node)
    graph.add_conditional_edges("respond_final", check_human_decision)

    # Agent 2
    graph.add_node("Agent2", action=second_agent_node)
    # Tools for Agent 2
    graph.add_node("tool_wikipedia", action=tool_node)
    graph.add_edge("tool_wikipedia", "Agent2")
    graph.add_conditional_edges("Agent2", determine_next_tool)

    graph.add_edge(END, END)
    return graph.compile()
