# %%
import math
import time
import threading
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
# 헬퍼
# ============================================================

HOME_JOINTS       = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
LEFT_DEFAULT_ORI  = [0, np.pi/2, 0]
RIGHT_DEFAULT_ORI = [0, np.pi/2, np.pi]

# 배치 기준 위치: 빨간(x<0)=오른팔, 파란(x>0)=왼팔
RIGHT_PLACE = [-0.3, 0.0, 1.0]
LEFT_PLACE  = [ 0.3, 0.0, 1.0]
Y_GAP = 0.08


def _wait_converge(arm='left', tol=0.003, timeout=10.0):
    """EE 위치 변화가 tol(m) 이하가 될 때까지 폴링."""
    t0 = time.time()
    prev = np.array(get_state(arm)['ee_position'])
    time.sleep(0.1)
    while time.time() - t0 < timeout:
        curr = np.array(get_state(arm)['ee_position'])
        if np.linalg.norm(curr - prev) < tol:
            return True
        prev = curr
        time.sleep(0.05)
    return False


# ============================================================
# [Setup] 환경 조회
# ============================================================

# %%
print("--- 환경 조회 ---")
env     = get_env()
objects = env['objects']
reds  = sorted([(k, v) for k, v in objects.items() if 'red'  in k])
blues = sorted([(k, v) for k, v in objects.items() if 'blue' in k])
print(f"빨간 큐브: {[k for k,_ in reds]}")
print(f"파란 큐브: {[k for k,_ in blues]}")


# ============================================================
# [동시 실행] 양팔 pick & place
#
# 충돌 방지 전략:
#   - pick/place 동작 영역: 오른팔 x<0, 왼팔 x>0 → 겹치지 않아 동시 진행 가능
#   - 홈 복귀(waist=0)는 양팔이 중앙 통과 → home_lock으로 한 팔씩 순차 진행
#   - IK 실패(ik_success=False) 시 0.5s 대기 후 최대 3회 재시도
# ============================================================

# %%
# 충돌 방지: 서버의 _check_trajectory_collision 에 위임
# IK 실패(궤적 충돌 포함) 시 1초 후 재시도


def _set_ee_retry(pos, ori, arm, retries=5, wait=1.0):
    """IK/궤적 충돌 실패 시 wait초 대기 후 재시도."""
    for attempt in range(retries):
        result = set_ee(pos, target_ori=ori, arm=arm)
        if result.get('ik_success', False):
            return result
        time.sleep(wait)

    return result


def _safe_home(arm):
    """홈 복귀 — 각 팔 독립 실행."""
    set_joint(HOME_JOINTS, arm=arm)
    _wait_converge(arm)


def _ts():
    return time.strftime("%H:%M:%S")

def _run_arm(pending, place_base, ori, arm, label):
    """pick & place 공통 루프: 접근 가능한 오브젝트 우선 처리, 완료 시 큐에서 제거."""
    from collections import deque
    queue = deque(pending)   # (name, info)
    place_idx = 0
    skip_count = 0

    while queue:
        # 한 바퀴 돌아도 처리된 것이 없으면 잠시 대기
        if skip_count >= len(queue):
            print(f"[{_ts()}][{label}] 처리 가능한 오브젝트 없음 → 0.5s 대기")
            time.sleep(0.5)
            skip_count = 0
            continue

        name, info = queue.popleft()
        p = info['pos']

        # approach 위치로 이동 시도 (IK 가능 여부 확인)
        result = set_ee([p[0], p[1], p[2]+0.10], target_ori=ori, arm=arm)
        if not result.get('ik_success', False):
            print(f"[{_ts()}][{label}] {name} 도달 불가 → 큐 뒤로 이동")
            queue.append((name, info))
            skip_count += 1
            continue

        skip_count = 0
        dp = [place_base[0], place_base[1] + place_idx * Y_GAP, place_base[2]]

        # pick: 그리퍼 열기 → approach 수렴 대기 → 파지 → 들어올리기 → home
        print(f"[{_ts()}][{label}] {name} 집기 | pos={np.round(p, 3)}")
        set_gripper(0.074, arm=arm); time.sleep(0.5)
        _wait_converge(arm)
        _set_ee_retry([p[0], p[1], p[2]+0.015], ori, arm); _wait_converge(arm)
        set_gripper(0.01, arm=arm); time.sleep(1.5)
        _set_ee_retry([p[0], p[1], p[2]+0.20],  ori, arm); _wait_converge(arm)
        _safe_home(arm)

        # place: 목표 위치로 배치 → 그리퍼 열기 → home
        print(f"[{_ts()}][{label}] {name} 배치 | dp={np.round(dp, 3)}")
        set_gripper(0.074, arm=arm); time.sleep(1.0)
        _safe_home(arm)

        place_idx += 1
        print(f"[{_ts()}][{label}] {name} 완료 ✓ (남은: {len(queue)}개)")


def _run_right():
    """오른팔: 빨간 큐브 pick & place (접근 가능 우선)."""
    _run_arm(reds, RIGHT_PLACE, RIGHT_DEFAULT_ORI, 'right', 'right')


def _run_left():
    """왼팔: 파란 큐브 pick & place (접근 가능 우선)."""
    _run_arm(blues, LEFT_PLACE, LEFT_DEFAULT_ORI, 'left', 'left')


# %%
print("--- 양팔 동시 실행 시작 ---")
t_right = threading.Thread(target=_run_right, name='right-arm')
t_left  = threading.Thread(target=_run_left,  name='left-arm')
t_right.start()
time.sleep(1.0)
t_left.start()
t_right.join()
t_left.join()
print("--- 완료 ---")

# %%
