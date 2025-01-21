# Multi Agent Project
This project shows a rough, basic implementation of a multi-agent system that uses different tools like browser, Wikipedia.

## Overview
- We have multiple Python files:
  - `agent_logic.py` defines our agent data model, memory, and the function to trigger the agent.
  - `tools.py` has DuckDuckGo and Wikipedia search tools, plus a final responder.
  - `workflow_builder.py` builds the LangChain-based graph connecting agents and tools.
  - `main.py` is the entry script. Run it after installing dependencies.

## Usage
1. `pip install -r requirements.txt`
2. `python main.py`
3. Answer any console prompts. That’s it, done.
