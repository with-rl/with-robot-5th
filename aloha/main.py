"""FastAPI server for MuJoCo ALOHA robot simulation with REST API control."""

import time
import threading
import uvicorn
from typing import Dict, Any, Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
from simulator import MujocoSimulator
import code_repository


# Server configuration
HOST = "0.0.0.0"  # Listen on all network interfaces
PORT = 8801       # API server port (different from robot to allow both to run)
VERSION = "0.0.1"

# FastAPI application instance
app = FastAPI(
    title="MuJoCo ALOHA Simulator API",
    description="Control ALOHA dual-arm robot via REST API",
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
    return {"name": "MuJoCo ALOHA Simulator", "version": VERSION, "status": "running"}


@app.get("/state")
def get_state() -> Dict[str, Any]:
    """Get current robot state including arm joints and end effector position."""
    ee_pos, ee_ori = simulator.get_ee_position()
    return {
        "timestamp": time.time(),
        "arm_joint_position": simulator.get_arm_joint_position().tolist(),
        "ee_position": ee_pos.tolist(),
        "ee_orientation": ee_ori.tolist(),
        "gripper_width": simulator.get_gripper_width(),
    }


@app.get("/env")
def get_environment() -> Dict[str, Any]:
    """Collect environment snapshot with object poses."""
    objects = simulator.get_object_positions()
    for obj in objects.values():
        obj['pos'] = obj['pos'].tolist()
        obj['ori'] = obj['ori'].tolist()
    return {
        "timestamp": time.time(),
        "objects": objects,
    }


class BimanualMoveRequest(BaseModel):
    target_pos: List[float]
    target_ori: List[float]
    dq_rel: List[float]       # dual quaternion 8값 (파지 직후 get_rel_pose로 획득)
    leader: str = 'left'
    follower: str = 'right'
    timeout: float = 10.0
    verbose: bool = False


class JointRequest(BaseModel):
    target: List[float]
    arm: str = 'left'

class EERequest(BaseModel):
    target_pos: List[float]
    target_ori: Optional[List[float]] = None
    arm: str = 'left'

class GripperRequest(BaseModel):
    width: float
    arm: str = 'left'


@app.get("/arm/state")
def get_arm_state(arm: str = 'left') -> Dict[str, Any]:
    """Get joint positions and EE pose for a specific arm."""
    ee_pos, ee_ori = simulator.get_ee_position(arm=arm)
    return {
        "arm": arm,
        "joint_position": simulator.get_arm_joint_position(arm=arm).tolist(),
        "ee_position": ee_pos.tolist(),
        "ee_orientation": ee_ori.tolist(),
        "gripper_width": simulator.get_gripper_width(arm=arm),
    }


@app.post("/arm/joint")
def set_arm_joint(req: JointRequest) -> Dict[str, Any]:
    """Set arm target joint positions (non-blocking)."""
    import numpy as np
    simulator.set_arm_target_joint(np.array(req.target), arm=req.arm)
    return {"status": "ok", "arm": req.arm, "target": req.target}


@app.post("/arm/ee")
def set_arm_ee(req: EERequest) -> Dict[str, Any]:
    """Set arm EE target position via IK (non-blocking). Returns IK result."""
    import numpy as np
    target_ori = np.array(req.target_ori) if req.target_ori is not None else None
    success, q = simulator.set_ee_target_position(np.array(req.target_pos), target_ori=target_ori, arm=req.arm)
    return {"status": "ok", "arm": req.arm, "ik_success": success, "joint_angles": q.tolist()}


@app.post("/gripper")
def set_gripper(req: GripperRequest) -> Dict[str, Any]:
    """Set gripper target width (non-blocking)."""
    simulator.set_target_gripper_width(req.width, arm=req.arm)
    return {"status": "ok", "arm": req.arm, "width": req.width}


@app.get("/bimanual/rel_pose")
def get_bimanual_rel_pose(leader: str = 'left', follower: str = 'right') -> Dict[str, Any]:
    """파지 직후 두 EE 간 상대 포즈(dq_rel) 계산. bimanual/move 호출 전에 먼저 호출."""
    dq_rel = simulator.get_rel_pose(leader=leader, follower=follower)
    return {"leader": leader, "follower": follower, "dq_rel": dq_rel.tolist()}


@app.post("/bimanual/move")
def bimanual_move(req: BimanualMoveRequest) -> Dict[str, Any]:
    """양팔이 강체를 유지하며 목표 포즈로 동시 이동 (blocking)."""
    import numpy as np
    success_leader, success_follower = simulator.bimanual_move_object(
        np.array(req.target_pos),
        np.array(req.target_ori),
        np.array(req.dq_rel),
        leader=req.leader,
        follower=req.follower,
        timeout=req.timeout,
        verbose=req.verbose,
    )
    return {"status": "ok", "leader_success": success_leader, "follower_success": success_follower}


@app.post("/send_action")
def receive_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Queue action for execution.

    Expected format:
        {
            "action": {
                "type": "run_code",
                "payload": {"code": "set_arm_target_joint([0, -0.96, 1.16, 0, -0.3, 0])"}
            }
        }
    """
    # Validate action format
    if "action" in payload and "type" in payload["action"] and "payload" in payload["action"]:
        print(payload["action"])
        RESULT = process_actions(payload["action"])
        return {"status": "success", "result": RESULT}
    
    return {"status": "error", "message": "Invalid action format"}


def main() -> None:
    """
    Start simulator and FastAPI server.

    Creates concurrent threads:
        1. Main thread: FastAPI uvicorn server
        2. Simulator thread: MuJoCo physics simulation with 3D viewer
    """
    # Start background threads (daemon=True ensures cleanup on exit)
    threading.Thread(target=run_simulator, daemon=True).start()

    # Display startup information
    print("\n" + "=" * 60)
    print(f"MuJoCo ALOHA Simulator API")
    print("=" * 60)
    print(f"Server: http://{HOST}:{PORT}")
    print(f"API docs: http://{HOST}:{PORT}/docs")
    print("=" * 60 + "\n")

    # Start FastAPI server (blocking call)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
