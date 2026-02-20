# %%
import requests
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_URL = "http://localhost:8801"

def run_step(code_str, print_output=True):
    """Helper function to send and execute code in the server's sandbox environment."""
    payload = {
        "action": {
            "type": "run_code",
            "payload": {"code": code_str}
        }
    }
    response = requests.post(f"{BASE_URL}/send_action", json=payload)
    result = response.json()
    if print_output:
        print(result)
    return result

# ==========================================
# [Preparation] Pick up object and move to handover position
# ==========================================

# %%
print("--- [Prep 1] Fetching cube position from environment ---")
env_data = requests.get(f"{BASE_URL}/env").json()
obj_pos = env_data['objects']['object_red_0']['pos']

# %%
print("--- [Prep 2] Left arm: Picking up object and moving to center ---")
run_step(f"pick_object({obj_pos}, arm='left', verbose=True)")

# %%
table_center_pos = [0.0, 0.0, 0.3]  # Hovering at table center (y=0.0)
run_step(f"set_ee_target_position({table_center_pos}, arm='left', timeout=10.0, verbose=True)")

# ==========================================
# [Handover] Cross Grasp Control
# ==========================================

# %%
print("--- [Handover] Updating reference Orientation/Position ---")
# Fetch the latest state of the left arm and the object for accurate approach calculations.
get_leff_ee_code = """
pos, ori = get_ee_position(arm='left')
RESULT['lef_ee_pos'] = pos
RESULT['lef_ee_ori'] = ori
"""
result = run_step(get_leff_ee_code, print_output=False)
lef_ee_pos = result['result']['lef_ee_pos']
lef_ee_ori = result['result']['lef_ee_ori']

# %%
# Use the object's real-time center axis to prevent collisions.
env_data = requests.get(f"{BASE_URL}/env").json()
obj_pos = env_data['objects']['object_red_0']['pos']

# %%
print("--- 1. Open right gripper ---")
run_step("set_target_gripper_width(0.074, arm='right', timeout=2.0)")

# %%
# [Core Idea] Pre-rotate the right wrist manually by 90 degrees to prevent IK Solver local minimums (Gimbal Lock).
print("--- 2. Rotate right wrist (Prepare for cross grasp) ---")
pre_rotate_code = f"""
pos, ori = get_ee_position(arm='right')
diff = ({lef_ee_ori[0]} + math.pi / 2) - ori[0]

q = get_arm_joint_position(arm='right')
q[5] += diff
set_arm_target_joint(q, arm='right', timeout=2.0)
"""
run_step(pre_rotate_code)

# %%
# Get the unique direction vector (downward) the left arm is pointing towards
left_rot = R.from_euler("xyz", lef_ee_ori).as_matrix()
left_down_dir = left_rot[:, 0]  # Local X-axis (direction the left gripper extends to)

# %%
print("--- 3. Approach along the object's direction vector ---")
# Wait at a point 10cm behind the object along the left arm's downward trajectory.
approach_pos = (np.array(obj_pos) + left_down_dir * 0.10).tolist()
run_step(f"set_ee_target_position({approach_pos}, arm='right', timeout=5.0)")

# %%
print("--- 4. Move to grasp position ---")
# Slide right in to overlap the object's exact center.
grasp_pos = (np.array(obj_pos)).tolist()
run_step(f"set_ee_target_position({grasp_pos}, arm='right', timeout=5.0)")

# %%
print("--- 5. Close right gripper ---")
run_step("set_target_gripper_width(0.015, arm='right', timeout=2.0)")

# %%
print("--- 6. Open left gripper ---")
run_step("set_target_gripper_width(0.074, arm='left', timeout=2.0)")

# %%
print("--- 7. Retract left arm to avoid collision ---")
# Retract cleanly by 15cm precisely opposite to the left arm's approach trajectory.
left_retract_pos = (np.array(lef_ee_pos) - left_down_dir * 0.15).tolist()
run_step(f"set_ee_target_position({left_retract_pos}, arm='left', timeout=5.0)")

# %%
print("--- 8. Return left arm to Home position ---")
run_step("set_arm_target_joint([0.0, -0.96, 1.16, 0.0, -0.3, 0.0], arm='left', timeout=5.0)")

# %%
print("--- 9. Retract right arm with the object (Handover complete) ---")
# Retreat 10cm back along the initial approach path to avoid collision.
lift_pos = (np.array(obj_pos) + left_down_dir * 0.10).tolist()
run_step(f"set_ee_target_position({lift_pos}, arm='right', timeout=5.0)")

# %%
print("--- 10. Return right arm to Home position (Holding object) ---")
run_step("set_arm_target_joint([0.0, -0.96, 1.16, 0.0, -0.3, 0.0], arm='right', timeout=5.0)")
print("✨ --- Handover Sequence Completed --- ✨")

# %%
