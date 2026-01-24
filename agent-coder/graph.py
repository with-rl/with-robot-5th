import os
import re
import json
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_fireworks import ChatFireworks
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Dict


# Load environment variables
load_dotenv()

ROBOT_URL = "http://127.0.0.1:8800"

# Initialize LLM
llm = ChatFireworks(
    model="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
    max_tokens=1000
)

# Load code repository once at module initialization
CODE_KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "robot", "code_knowledge.md")
def load_code_knowledge():
    """Load robot control API documentation."""
    try:
        with open(CODE_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "# Error: code_knowledge.md not found"
# Load once at module level
CODE_KNOWLEDGE = load_code_knowledge()


class State(TypedDict):
    """AI Agent conversation state."""
    generated_code: str # Generated code
    exec_result: dict # Execution result
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]  # Conversation history


def plan_node(state: State) -> State:
    """Create Robot Control code."""
    try:
        payload = {"action": {
                        "type": "run_code",
                        "payload": {
                            "code": """objects = get_object_positions()
RESULT["objects"] = objects"""
                        }
                    }}
        response = requests.post(f"{ROBOT_URL}/send_action", json=payload)
        objects = response.json()["result"]["objects"]
        objects_str = json.dumps(objects, indent=2)

        system_prompt = f"""You are a helpful robot assistant.

Generate Python code based on the user's command using the API below.

## Scene Context (Available Objects)
{objects_str}

## Rules
- Return ONLY executable Python code in a single generic block used markdown.
- NO imports allowed (`time`, `math`, `list` are pre-loaded).
- Use `RESULT` dict for return values.
- Be concise and natural.
- When referring to objects, use the exact names from Scene Context.

## API Reference
{CODE_KNOWLEDGE}
```"""

        generated_code = llm.invoke([
                                        SystemMessage(content=system_prompt)
                                    ] + state["messages"])
        # Extract code using regex
        code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", generated_code.content, re.DOTALL)
        if code_match:
            generated_code = code_match.group(1).strip()
        
        return {"generated_code": generated_code}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"messages": [traceback.format_exc()], "generated_code": None}


def exec_node(state: State) -> State:
    """Execute Robot Control code."""
    try:
        generated_code = state["generated_code"]
        
        payload = {"action": {
                "type": "run_code",
                "payload": {
                    "code": generated_code
                }
            }}
        exec_result = requests.post(f"{ROBOT_URL}/send_action", json=payload).json()
        
        return {"exec_result": exec_result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"messages": [traceback.format_exc()], "exec_result": None}


def create_graph(checkpointer=None):
    """Create chat graph with optional checkpointer for memory."""
    graph_builder = (
        StateGraph(State)
        .add_node("plan", plan_node)
        .add_node("exec", exec_node)
        .add_edge(START, "plan")
        .add_edge("plan", "exec")
        .add_edge("exec", END)
    )
    
    if checkpointer:
        return graph_builder.compile(checkpointer=checkpointer)
    else:
        return graph_builder.compile()


__all__ = ["graph", "create_graph"]
