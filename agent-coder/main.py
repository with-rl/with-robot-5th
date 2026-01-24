"""FastAPI server for LLM Agent with REST API control."""

import json
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
from graph import create_graph
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


# Server configuration
HOST = "0.0.0.0"  # Listen on all network interfaces
PORT = 8900       # API server port
VERSION = "0.0.1"

# FastAPI application instance
app = FastAPI(
    title="LLM Agent API",
    description="Generate Panda-Omron control code",
    version=VERSION
)

# Create graphs with memory
code_memory = MemorySaver()
code_graph = create_graph(code_memory)


@app.get("/")
def get_ui() -> HTMLResponse:
    """Get server info."""
    return HTMLResponse(content=open("ui.html", "r").read())


@app.post("/llm_command")
def llm_command(request: dict):
    """
    Receives natural language commands and generates/executes robot control code.

    Request format:
        {
            "command": "Move in a square pattern"
        }

    Response format:
        {
            "status": "success",
            "user_command": "...",
            "generated_code": "..."
        }
    """
    try:
        user_command = request.get("command", "")

        if not user_command:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": "No command provided"}
            )
        
        # Generate session ID if not provided
        session_id = request.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())

        # Create thread configuration for checkpointer
        config = {"configurable": {"thread_id": session_id}}

        # Run the chat graph with memory
        result = code_graph.invoke(
            {"messages": [HumanMessage(content=user_command)]},
            config=config
        )

        # Extract the AI response
        generated_code = result["generated_code"]
        exec_result = result["exec_result"]

        # Return response with session ID
        return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "success",
                    "user_command": user_command,
                    "generated_code": generated_code,
                    "exec_result": json.dumps(exec_result, ensure_ascii=False, indent=2),
                    "session_id": session_id
                }
            )

    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        )


def main() -> None:
    """
    Start FastAPI server.
    """
    # Display startup information
    print(f"\n{"="*60}")
    print(f"LLM Agent API")
    print(f"{"="*60}")
    print(f"Server: http://{HOST}:{PORT}")
    print(f"API docs: http://{HOST}:{PORT}/docs")
    print(f"{"="*60}\n")

    # Start FastAPI server (blocking call)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
