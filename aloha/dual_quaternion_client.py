# %%
import math
import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

BASE_URL = "http://localhost:8801"


# ============================================================
# 직접 제어 함수 (non-blocking)
# ============================================================

def set_joint(target, arm='left'):
    """관절 각도 설정."""
    return requests.post(f"{BASE_URL}/arm/joint", json={"target": list(target), "arm": arm}).json()

def set_ee(target_pos, target_ori=None, arm='left'):
    """EE 위치/방향 설정 (IK). IK 성공 여부 반환."""
    body = {"target_pos": list(target_pos), "arm": arm}
    if target_ori is not None:
        body["target_ori"] = list(target_ori)
    return requests.post(f"{BASE_URL}/arm/ee", json=body).json()

def set_gripper(width, arm='left'):
    """그리퍼 너비 설정."""
    return requests.post(f"{BASE_URL}/gripper", json={"width": width, "arm": arm}).json()

def get_state(arm='left'):
    """관절 위치 + EE 포즈 조회."""
    return requests.get(f"{BASE_URL}/arm/state", params={"arm": arm}).json()

def get_env():
    """환경 내 오브젝트 위치 조회."""
    return requests.get(f"{BASE_URL}/env").json()

def get_rel_pose(leader='left', follower='right'):
    """파지 직후 두 EE 간 상대 포즈(dq_rel) 계산. bimanual_move 호출 전에 먼저 호출."""
    return requests.get(f"{BASE_URL}/bimanual/rel_pose", params={"leader": leader, "follower": follower}).json()

def bimanual_move(target_pos, target_ori, dq_rel, leader='left', follower='right', timeout=10.0, verbose=False):
    """양팔이 강체를 유지하며 목표 포즈로 동시 이동 (blocking)."""
    return requests.post(f"{BASE_URL}/bimanual/move", json={
        "target_pos": list(target_pos),
        "target_ori": list(target_ori),
        "dq_rel": list(dq_rel),
        "leader": leader,
        "follower": follower,
        "timeout": timeout,
        "verbose": verbose,
    }).json()


# ============================================================
# [Grasp] 오른팔 파지
# ============================================================

# %%
print("\n--- [오른팔 파지] 현재 상태 조회 ---")
right = get_state('right')
right_ee_pos = right['ee_position']
right_ee_ori = right['ee_orientation']
print(f"Right EE pos={np.round(right_ee_pos, 3)}, ori={np.round(right_ee_ori, 3)}")

env = get_env()
obj_pos = env['objects']['object_red_0']['pos']
print(f"Object pos={obj_pos}")

# %%
print("--- 1. 오른팔 그리퍼 열기 ---")
set_gripper(0.074, arm='right')

# %%
print("--- 2. 오른팔: 오브젝트 방향으로 접근 ---")
target_ori = np.array([0, np.pi/2, np.pi])
approach_pos = np.array([obj_pos[0] + 0.15, obj_pos[1], obj_pos[2] + 0.1])
print(set_ee(approach_pos, target_ori=target_ori, arm='right'))

# %%
print("--- 3. 오른팔: 파지 위치로 이동 ---")
target_ori = np.array([0, np.pi/2, np.pi])
approach_pos = np.array([obj_pos[0] + 0.15, obj_pos[1], obj_pos[2]])
print(set_ee(approach_pos, target_ori=target_ori, arm='right'))

# %%
print("--- 4. 오른팔 그리퍼 닫기 ---")
set_gripper(0.015, arm='right')


# ============================================================
# [Grasp] 왼팔 파지
# ============================================================

# %%
print("\n--- [왼팔 파지] 현재 상태 조회 ---")
left = get_state('left')
lef_ee_pos = left['ee_position']
lef_ee_ori = left['ee_orientation']
print(f"Left EE pos={np.round(lef_ee_pos, 3)}, ori={np.round(lef_ee_ori, 3)}")

env = get_env()
obj_pos = env['objects']['object_red_0']['pos']
print(f"Object pos={obj_pos}")

# %%
print("--- 1. 왼팔 그리퍼 열기 ---")
set_gripper(0.074, arm='left')

# %%
print("--- 2. 왼팔: 접근 위치로 이동 ---")
target_ori = np.array([0, np.pi/2, 0])
approach_pos = np.array([obj_pos[0] - 0.15, obj_pos[1], obj_pos[2] + 0.1])
print(set_ee(approach_pos, target_ori=target_ori, arm='left'))

# %%
print("--- 3. 왼팔: 파지 위치로 이동 ---")
target_ori = np.array([0, np.pi/2, 0])
approach_pos = np.array([obj_pos[0] - 0.15, obj_pos[1], obj_pos[2]])
print(set_ee(approach_pos, target_ori=target_ori, arm='left'))

# %%
print("--- 4. 왼팔 그리퍼 닫기 ---")
set_gripper(0.015, arm='left')


# ============================================================
# [Bimanual] 양팔 동시 이동 (강체 유지)
# ============================================================

# %%
# 양팔이 물체를 잡은 직후 상대 포즈 측정 (한 번만 호출)
print("\n--- [Bimanual] 상대 포즈 측정 ---")
rel = get_rel_pose(leader='left', follower='right')
dq_rel = rel['dq_rel']
print(f"dq_rel={np.round(dq_rel, 4)}")

# %%
# leader(왼팔) 목표 포즈 지정 → follower(오른팔)는 dq_rel 유지하며 자동 계산
print("--- [Bimanual] 양팔 동시 이동: z+0.1 들어올리기 ---")
left = get_state('left')
lift_pos = np.array(left['ee_position']) + np.array([0, 0, 0.1])
lift_ori = np.array(left['ee_orientation'])
print(bimanual_move(lift_pos, lift_ori, dq_rel, leader='left', follower='right', timeout=10.0, verbose=True))

# %%
print("--- [Bimanual] 양팔 동시 이동: 작은 원 그리기 (YZ 평면) ---")
left = get_state('left')
center  = np.array(left['ee_position'])
base_ori = np.array(left['ee_orientation'])

radius = 0.05   # 원 반지름 (m)
steps  = 16     # 분할 수

for i in range(steps + 1):
    angle = 2 * math.pi * i / steps
    target_pos = center + np.array([0, radius * math.cos(angle), radius * math.sin(angle)])
    print(f"  step {i:2d} | angle={math.degrees(angle):5.1f}° | pos={np.round(target_pos, 3)}")
    result = bimanual_move(target_pos, base_ori, dq_rel, leader='left', follower='right', timeout=5.0)
    print(f"         → {result}")

# %%
print("--- [Bimanual] 양팔 동시 이동: 작은 원 그리기 (XY 평면, 수평) ---")
left = get_state('left')
center   = np.array(left['ee_position'])
base_ori = np.array(left['ee_orientation'])

radius = 0.05   # 원 반지름 (m)
steps  = 16     # 분할 수

for i in range(steps + 1):
    angle = 2 * math.pi * i / steps
    target_pos = center + np.array([radius * math.cos(angle), radius * math.sin(angle), 0])
    print(f"  step {i:2d} | angle={math.degrees(angle):5.1f}° | pos={np.round(target_pos, 3)}")
    result = bimanual_move(target_pos, base_ori, dq_rel, leader='left', follower='right', timeout=5.0)
    print(f"         → {result}")

# %%
