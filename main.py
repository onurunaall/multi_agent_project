"""
main.py
Entry point for running the multi-agent system.
"""

from workflow_builder import (
    build_single_agent_flow,
    build_multi_agent_flow,
    AgentState
)


def main():
    # Sample user question and conversation
    user_query = "Who died on September 9, 2024?"
    chat_log = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi, what's your question?"},
        {"role": "user", "content": "I'd like some info."},
        {"role": "assistant", "content": "Sure, what's on your mind?"}
    ]

    initial_state: AgentState = {
        "user_q": user_query,
        "chat_history": chat_log,
        "lst_res": [],
        "output": {},
        "attempt_count": 0  # Starting with zero attempts
    }

    # Single Agent Flow
    print("=== Single Agent Flow ===")
    single_flow = build_single_agent_flow()
    result_single = single_flow.invoke(input=initial_state.copy())
    print("Single-agent final output:", result_single.get("output", {}))

    # Multi Agent Flow
    print("\n=== Multi-Agent Flow ===")
    multi_flow = build_multi_agent_flow()
    result_multi = multi_flow.invoke(input=initial_state.copy())
    print("Multi-agent final output:", result_multi.get("output", {}))


if __name__ == "__main__":
    main()
