"""Sandboxed code execution layer for ALOHA arm control with waiting logic.

This code repository is designed for the ALOHA robot which has:
- Two fixed arms (left and right) mounted on a table (no mobile base)
- Each arm has 6 DOF (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate)
- Each arm has a 2-finger parallel gripper

Currently implemented: Left arm only (single arm control)
"""

import time
import math
import numpy as np
from typing import Optional, List, Tuple, Callable, Dict, Any
from simulator import MujocoSimulator

# Simulator instance injected by main.py at startup
simulator: MujocoSimulator = None


def _wait_for_convergence(
    get_pos_diff_fn: Callable[[], np.ndarray],
    get_vel_fn: Callable[[], np.ndarray],
    pos_threshold: float,
    vel_threshold: float,
    timeout: float = 10.0,
    stable_frames: int = 5,
    verbose: bool = False
) -> bool:
    """Wait for position and velocity convergence with stability check."""
    start_time = time.time()
    stable_count = 0
    iterations = 0
    pos_error = float('inf')
    vel_error = float('inf')

    while time.time() - start_time < timeout:
        pos_diff = get_pos_diff_fn()
        velocity = get_vel_fn()

        pos_error = np.linalg.norm(pos_diff)
        vel_error = np.linalg.norm(velocity)

        # Check both position and velocity convergence
        if pos_error < pos_threshold and vel_error < vel_threshold:
            stable_count += 1
            if stable_count >= stable_frames:
                if verbose:
                    print(f"Converged after {time.time() - start_time:.2f}s ({iterations} iterations)")
                return True
        else:
            stable_count = 0  # Reset if not stable

        iterations += 1

        # Adaptive sleep: longer when far, shorter when close
        if pos_error > pos_threshold * 3:
            time.sleep(0.1)
        elif pos_error > pos_threshold * 1.5:
            time.sleep(0.05)
        else:
            time.sleep(0.02)

    if verbose:
        print(f"Timeout after {timeout}s (pos_error={pos_error:.4f}, vel_error={vel_error:.4f})")
    return False


# ============================================================
# Arm Joint Control Functions
# ============================================================

def get_arm_joint_position(arm: str = 'left') -> List[float]:
    """Get current arm joint positions [j1~j6] in radians."""
    pos = simulator.get_arm_joint_position(arm=arm).tolist()
    return pos


def set_arm_target_joint(
    arm_target_position: List[float],
    arm: str = 'left',
    timeout: float = 10.0,
    verbose: bool = False
) -> bool:
    """Set arm target joint positions [j1~j6] in radians."""
    # Update arm target position immediately (non-blocking)
    simulator.set_arm_target_joint(arm_target_position, arm=arm)

    success = True
    if success and timeout > 0:
        converged = _wait_for_convergence(
            lambda: simulator.get_arm_joint_diff(arm=arm),
            lambda: simulator.get_arm_joint_velocity(arm=arm),
            pos_threshold=0.1,
            vel_threshold=0.1,  # ~0.1 rad/s
            timeout=timeout,
            stable_frames=5,
            verbose=verbose
        )
        success = converged
    return success


# ============================================================
# End Effector Control Functions
# ============================================================

def get_ee_position(arm: str = 'left') -> Tuple[List[float], List[float]]:
    """Get current end effector pose as tuple: (position, orientation)."""
    pos, ori = simulator.get_ee_position(arm=arm)
    return pos.tolist(), ori.tolist()


def set_ee_target_position(
    target_pos: List[float],
    target_ori: Optional[List[float]] = None,
    arm: str = 'left',
    timeout: float = 10.0,
    verbose: bool = False
) -> bool:
    """Set end effector target position in world frame."""
    success, joint_angles = simulator.set_ee_target_position(target_pos, target_ori=target_ori, arm=arm)

    if success and timeout > 0:
        converged = _wait_for_convergence(
            lambda: simulator.get_arm_joint_diff(arm=arm),
            lambda: simulator.get_arm_joint_velocity(arm=arm),
            pos_threshold=0.1,
            vel_threshold=0.1,
            timeout=timeout,
            stable_frames=5,
            verbose=verbose
        )
        success = converged
    return success


# ============================================================
# Gripper Control Functions
# ============================================================

def get_gripper_width(arm: str = 'left') -> float:
    """Get current gripper width in meters."""
    return simulator.get_gripper_width(arm=arm)


def set_target_gripper_width(
    target_width: float,
    arm: str = 'left',
    timeout: float = 10.0,
    verbose: bool = False
) -> bool:
    """Set gripper target width."""
    simulator.set_target_gripper_width(target_width, arm=arm)

    success = True
    if success and timeout > 0:
        converged = _wait_for_convergence(
            lambda: [simulator.get_gripper_width_diff(arm=arm)],
            lambda: [0.0],  # Gripper velocity not implemented, just skip Check
            pos_threshold=0.01,
            vel_threshold=0.01,
            timeout=timeout,
            stable_frames=5,
            verbose=verbose
        )
        time.sleep(1.0)  # Wait for gripper to reach target width
        success = converged
    return success


# ============================================================
# Pick & Place Functions
# ============================================================

def pick_object(
    object_pos: List[float],
    arm: str = 'left',
    approach_height: float = 0.1,
    lift_height: float = 0.2,
    return_to_home: bool = True,
    timeout: float = 10.0,
    verbose: bool = False
) -> bool:
    """Pick up an object at the specified position."""
    return simulator.pick_object(
        np.array(object_pos), 
        arm=arm,
        approach_height=approach_height, 
        lift_height=lift_height, 
        return_to_home=return_to_home, 
        timeout=timeout, 
        verbose=verbose
    )


def place_object(
    place_pos: List[float],
    arm: str = 'left',
    approach_height: float = 0.2,
    retract_height: float = 0.3,
    return_to_home: bool = True,
    timeout: float = 10.0,
    verbose: bool = False
) -> bool:
    """Place an object at the specified position."""
    return simulator.place_object(
        np.array(place_pos), 
        arm=arm,
        approach_height=approach_height, 
        retract_height=retract_height, 
        return_to_home=return_to_home, 
        timeout=timeout, 
        verbose=verbose
    )


# ============================================================
# Code Execution
# ============================================================

def exec_code(code: str) -> Optional[Dict[str, Any]]:
    """
    Execute user code in sandboxed environment.

    Available functions:
        - get_arm_joint_position() -> [j1~j6] (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate)
        - set_arm_target_joint(arm_target_position, timeout, verbose)
        - get_ee_position() -> (position, orientation) where position=[x,y,z], orientation=[roll,pitch,yaw]
        - set_ee_target_position(target_pos, timeout, verbose)
        - get_gripper_width() -> current gripper width in meters
        - set_target_gripper_width(target_width, timeout, verbose)
        - pick_object(object_pos, approach_height, lift_height, return_to_home, timeout, verbose)
        - place_object(place_pos, approach_height, retract_height, return_to_home, timeout, verbose)
        
    Returns:
        Dict containing RESULT variable if set in executed code
    """
    # Define sandboxed environment with limited access
    safe_globals = {
        "__builtins__": {
            "print": print,
            "range": range,
            "float": float,
            "list": list,
            "time": time,
            "math": math,
            "PI": np.pi
        },
        "RESULT": {},
        "get_arm_joint_position": get_arm_joint_position,
        "set_arm_target_joint": set_arm_target_joint,
        "get_ee_position": get_ee_position,
        "set_ee_target_position": set_ee_target_position,
        "get_gripper_width": get_gripper_width,
        "set_target_gripper_width": set_target_gripper_width,
        "pick_object": pick_object,
        "place_object": place_object,
    }
    exec(code, safe_globals)
    return safe_globals.get("RESULT")
