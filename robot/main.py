"""FastAPI server for MuJoCo robot simulation with REST API control."""

import time
import queue
import threading
import uvicorn
from typing import Dict, Any, Optional
import cv2
import io
import base64
import numpy as np
from fastapi import FastAPI, Response, status
from fastapi.responses import StreamingResponse
from simulator import MujocoSimulator
import code_repository


# Server configuration
HOST = "0.0.0.0"  # Listen on all network interfaces
PORT = 8800       # API server port
VERSION = "0.0.1"

# FastAPI application instance
app = FastAPI(
    title="MuJoCo Robot Simulator API",
    description="Control Panda-Omron mobile robot via REST API",
    version=VERSION
)

# Create simulator instance and inject into code_repository
simulator = MujocoSimulator()
code_repository.simulator = simulator


def process_actions(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process action."""
    RESULT = {}
    if action["type"] == "run_code":
        code_str = action["payload"].get("code")
        try:
            RESULT = code_repository.exec_code(code_str)
            print(f"Code execution completed: {RESULT}")
        except Exception as e:
            # Log errors without crashing the simulator
            print(f"\n[EXECUTION ERROR]")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {e}")
            import traceback
            print(f"\n[TRACEBACK]")
            traceback.print_exc()
    print("=" * 60 + "\n")
    return RESULT


def run_simulator() -> None:
    """Run MuJoCo simulator in background thread."""
    simulator.run()


@app.get("/")
def read_root() -> Dict[str, str]:
    """Get server info."""
    return {"name": "MuJoCo Robot Simulator", "version": VERSION, "status": "running"}


import os

@app.on_event("shutdown")
def shutdown_event():
    """Triggered when the server exits (e.g., Ctrl+C), kills simulator thread."""
    print("Closing simulator thread...")
    try:
        simulator.close()
    except:
        pass
    finally:
        # Forcefully terminate to prevent MuJoCo's macOS viewer from hanging
        os._exit(0)


@app.get("/env")
def get_environment():
    """Collect environment snapshot with object poses and robot state."""
    objects = simulator.get_object_positions()
    for obj in objects.values():
        obj['pos'] = obj['pos'].tolist()
        obj['ori'] = obj['ori'].tolist()
    return {
        "timestamp": time.time(),
        "objects": objects,
    }


@app.get("/camera")
def get_camera_view(camera: str = "robot0_head_camera"):
    """Get an RGB-D image (RGB and Depth) from the specified camera."""
    rgb, depth = simulator.get_camera_rgbd(camera)
    if rgb is None or depth is None:
        return {"status": "error", "message": "Camera or renderer not available"}
    
    # 1. Encode RGB to JPEG (Base64)
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success_rgb, encoded_rgb = cv2.imencode('.jpg', img_bgr)
    if not success_rgb:
        return {"status": "error", "message": "Failed to encode RGB image"}
    rgb_base64 = base64.b64encode(encoded_rgb).decode('utf-8')
    
    # 2. Encode Depth to PNG (16-bit)
    # Depth is in meters (float32). Convert to millimeters (uint16) to preserve precision in PNG format.
    depth_mm = (depth * 1000).astype(np.uint16)
    success_depth, encoded_depth = cv2.imencode('.png', depth_mm)
    if not success_depth:
        return {"status": "error", "message": "Failed to encode Depth image"}
    depth_base64 = base64.b64encode(encoded_depth).decode('utf-8')
        
    # Return as JSON so both images can be transmitted together
    return {
        "status": "success",
        "camera": camera,
        "format": {
            "rgb": "image/jpeg",
            "depth": "image/png; mode=16bit_mm"
        },
        "rgb_base64": rgb_base64,
        "depth_base64": depth_base64
    }


@app.post("/send_action")
def receive_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Queue action for execution.

    Expected format:
        {
            "action": {
                "type": "run_code",
                "payload": {"code": "get_mobile_target_joint([0, 0, PI])"}
            }
        }
    """
    # Validate action format
    if "action" in payload and "type" in payload["action"] and "payload" in payload["action"]:
        RESULT = process_actions(payload["action"])
        return {"status": "success", "result": RESULT}
    
    return {"status": "error", "message": "Invalid action format"}


def main() -> None:
    """
    Start simulator and FastAPI server.

    Creates three concurrent threads:
        1. Main thread: FastAPI uvicorn server
        2. Simulator thread: MuJoCo physics simulation with 3D viewer
    """
    # Start background threads (daemon=True ensures cleanup on exit)
    threading.Thread(target=run_simulator, daemon=True).start()

    # Display startup information
    print("\n" + "=" * 60)
    print(f"MuJoCo Robot Simulator API")
    print("=" * 60)
    print(f"Server: http://{HOST}:{PORT}")
    print(f"API docs: http://{HOST}:{PORT}/docs")
    print("=" * 60 + "\n")

    # Start FastAPI server (blocking call)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
