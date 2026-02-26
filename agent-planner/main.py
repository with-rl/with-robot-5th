from __future__ import annotations

import base64
import io
import json
import os

import uvicorn
from dotenv import load_dotenv
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, Response

from src.config import config
from src.executor import TaskExecutor
from src.graph import create_graph
from src.state import make_state

url = "http://127.0.0.1:8800"

# Server configuration
HOST = "0.0.0.0"  # Listen on all network interfaces
PORT = 8900  # API server port
VERSION = "0.0.1"

# FastAPI application instance
app = FastAPI(
    title="LLM Agent API",
    description="Generate Panda-Omron control code",
    version=VERSION,
)

# Load environment variables
load_dotenv()

# Initialize ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Load configuration and initialize components
graph = create_graph(config=config)
task_executor = TaskExecutor()


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
                content={"status": "error", "message": "No command provided"},
            )

        state = make_state(user_query=user_command, config=config, url=url)

        final_state = graph.invoke(state)
        task_outputs = final_state["tasks"]["task_outputs"]

        print("*" * 40)
        print(task_outputs)
        print("*" * 40)

        results = task_executor.execute(task_outputs)

        generated_code = json.dumps(results, ensure_ascii=False, indent=2)

        # Return response with session ID
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "user_command": user_command,
                "generated_code": generated_code,
            },
        )

    except Exception as e:
        import traceback

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Convert speech to text using ElevenLabs STT.
    Accepts audio file and returns transcribed Korean text.
    """
    try:
        audio_bytes = await audio.read()

        # Use ElevenLabs STT
        transcription = elevenlabs_client.speech_to_text.convert(
            file=audio_bytes, model_id="scribe_v1", language_code="ko"
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "text": transcription.text},
        )
    except Exception as e:
        import traceback

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/tts")
async def text_to_speech(request: dict):
    """
    Convert text to speech using ElevenLabs TTS.
    Returns audio as base64 encoded string.
    """
    try:
        text = request.get("text", "")

        if not text:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": "No text provided"},
            )

        # Generate speech using ElevenLabs
        # Using a multilingual voice that supports Korean
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id="XB0fDUnXU5powFXDhCwa",  # Charlotte - multilingual voice
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        # Collect audio bytes from generator
        audio_bytes = b"".join(audio_generator)

        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "audio": audio_base64},
        )
    except Exception as e:
        import traceback

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    # Display startup information
    print("\n" + "=" * 60)
    print(f"LLM Agent API")
    print("=" * 60)
    print(f"Server: http://{HOST}:{PORT}")
    print(f"API docs: http://{HOST}:{PORT}/docs")
    print("=" * 60 + "\n")

    # Start FastAPI server (blocking call)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
