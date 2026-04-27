# %% [markdown]
"""
# Dual Quaternion 강체 유지 원리 튜토리얼

이 튜토리얼은 양팔 로봇이 물체를 함께 쥐고 이동할 때
**어떻게 물체 형태를 유지하는지**를 수학적으로 설명합니다.

핵심 공식:
    dq_follower = dq_leader ⊗ dq_rel

- dq_leader  : 왼팔(leader)의 현재 포즈
- dq_rel     : 파지 순간 측정한 두 팔 사이 상대 포즈 (고정값)
- dq_follower: 오른팔(follower)이 가야 할 포즈 (자동 계산)
"""

# %% [markdown]
"""
## 0. 라이브러리 준비
"""

# %%
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("라이브러리 로드 완료")

# %% [markdown]
"""
## 1. 쿼터니언(Quaternion)이란?

로봇 팔의 **방향(회전)** 을 표현하는 방법입니다.
4개의 숫자 [w, x, y, z] 로 3D 회전을 나타냅니다.

- 오일러 각도(roll, pitch, yaw)는 직관적이지만 "짐벌 락" 문제 발생
- 쿼터니언은 이 문제를 해결한 수학적 표현 방식

[w, x, y, z] 에서:
- w ≈ 1 이면 거의 회전 없음 (단위 쿼터니언)
- (x, y, z) 는 회전축 방향
"""

# %%
# 예시: Z축 기준 90도 회전
angle_deg = 90
angle_rad = np.radians(angle_deg)

# scipy를 이용해 쿼터니언 계산
rot = R.from_euler('z', angle_rad)
q_scipy = rot.as_quat()  # scipy 형식: [x, y, z, w]

# 우리 코드에서는 [w, x, y, z] 순서를 사용
q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

print(f"Z축 {angle_deg}도 회전 쿼터니언")
print(f"  [w, x, y, z] = {np.round(q, 4)}")
print(f"  w ≈ cos(45°) = {np.round(np.cos(np.radians(45)), 4)}")
print(f"  z ≈ sin(45°) = {np.round(np.sin(np.radians(45)), 4)}")
print(f"  → 크기(norm) = {np.round(np.linalg.norm(q), 4)} (항상 1이어야 함)")

# %% [markdown]
"""
## 1-1. 오일러 → 쿼터니언 직접 계산 (수식)

scipy 없이 수식으로 직접 변환합니다.
XYZ 순서(roll→pitch→yaw)로 각 축 회전을 합성하는 공식입니다.

각 축의 단위 쿼터니언:
- X축 roll  θ : [cos(θ/2),  sin(θ/2),  0,         0        ]
- Y축 pitch θ : [cos(θ/2),  0,         sin(θ/2),  0        ]
- Z축 yaw   θ : [cos(θ/2),  0,         0,         sin(θ/2) ]

XYZ 순서 합성: q = q_x ⊗ q_y ⊗ q_z
"""

# %%
def euler_to_quat(roll, pitch, yaw):
    """
    오일러 각도(XYZ 순서) → 쿼터니언 [w, x, y, z]

    Args:
        roll  : X축 회전 (라디안)
        pitch : Y축 회전 (라디안)
        yaw   : Z축 회전 (라디안)
    """
    cr, sr = np.cos(roll  / 2), np.sin(roll  / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw   / 2), np.sin(yaw   / 2)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy

    return np.array([w, x, y, z])


# 검증 1: Z축 90도 (앞서 scipy로 계산한 결과와 비교)
q_direct = euler_to_quat(0, 0, np.pi/2)
print("오일러 → 쿼터니언 직접 계산 검증")
print(f"  Z축 90° (직접 계산) = {np.round(q_direct, 4)}")
print(f"  Z축 90° (scipy)    = {np.round(q, 4)}")
print(f"  차이(norm)         = {np.round(np.linalg.norm(q_direct - q), 8)}\n")

# 검증 2: 다양한 각도
test_cases = [
    (np.pi/2, 0,       0,       "X축 90°"),
    (0,       np.pi/2, 0,       "Y축 90°"),
    (0,       0,       np.pi/2, "Z축 90°"),
    (np.pi/4, np.pi/4, np.pi/4, "XYZ 45°"),
]

print(f"{'케이스':<12} {'직접 계산':>36}  {'scipy':>36}  일치")
for roll, pitch, yaw, label in test_cases:
    q_d = euler_to_quat(roll, pitch, yaw)
    q_s_raw = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
    q_s = np.array([q_s_raw[3], q_s_raw[0], q_s_raw[1], q_s_raw[2]])
    match = np.allclose(q_d, q_s, atol=1e-6)
    print(f"  {label:<10} {str(np.round(q_d, 4)):>36}  {str(np.round(q_s, 4)):>36}  {'✓' if match else '✗'}")

# %% [markdown]
"""
## 2. 쿼터니언 곱(Quaternion Multiplication)

두 회전을 합성할 때 쿼터니언끼리 곱합니다.
행렬 곱과 비슷하지만 **순서가 중요** 합니다 (q1 * q2 ≠ q2 * q1).
"""

# %%
def quat_mul(q1, q2):
    """
    두 쿼터니언을 곱합니다.
    [w, x, y, z] 형식 입력.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,   # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,   # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,   # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,   # z
    ])

# 테스트: X축 90도 회전 후 Z축 90도 회전
q_x90 = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])  # X축 90도
q_z90 = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])  # Z축 90도

q_xz = quat_mul(q_x90, q_z90)  # X 먼저, 그 다음 Z

print("쿼터니언 곱 예시: X90° 회전 후 Z90° 회전")
print(f"  q_x90 = {np.round(q_x90, 4)}")
print(f"  q_z90 = {np.round(q_z90, 4)}")
print(f"  q_x90 * q_z90 = {np.round(q_xz, 4)}")
print(f"  결과 크기 = {np.round(np.linalg.norm(q_xz), 4)} (여전히 1)")

# %% [markdown]
"""
## 3. 듀얼 쿼터니언(Dual Quaternion)이란?

쿼터니언은 **회전만** 표현합니다.
**위치(translation)까지 함께** 표현하려면 듀얼 쿼터니언을 사용합니다.

```
DQ = [q_r | q_d]
      ↑       ↑
   회전 쿼터니언   위치 정보가 담긴 듀얼 파트
   (4개)         (4개) → 총 8개 숫자
```

q_d 계산법:
    q_d = 0.5 * [0, tx, ty, tz] * q_r

여기서 [0, tx, ty, tz]는 위치 벡터를 쿼터니언 형태로 표현한 것입니다.
"""

# %%
def pose_to_dq(position, euler_xyz):
    """
    위치(position)와 오일러 각도(euler_xyz) → 듀얼 쿼터니언

    Args:
        position  : [x, y, z] 위치 벡터
        euler_xyz : [roll, pitch, yaw] 오일러 각도 (라디안)

    Returns:
        dq : 8차원 듀얼 쿼터니언 [q_r | q_d]
    """
    # 1단계: 오일러 각도 → 쿼터니언 (직접 계산)
    q_r = euler_to_quat(*euler_xyz)

    # 2단계: 위치 벡터를 쿼터니언 형태로 표현 (w=0인 순수 쿼터니언)
    t_quat = np.array([0.0, position[0], position[1], position[2]])

    # 3단계: q_d = 0.5 * t_quat * q_r (위치 정보를 DQ에 인코딩)
    q_d = 0.5 * quat_mul(t_quat, q_r)

    return np.concatenate([q_r, q_d])


# 예시: 위치 (1.0, 0.5, 0.3), 방향 (0, 0, 0)
pos = np.array([1.0, 0.5, 0.3])
ori = np.array([0.0, 0.0, 0.0])   # 회전 없음

dq = pose_to_dq(pos, ori)

print("포즈 → 듀얼 쿼터니언 변환 예시")
print(f"  입력 위치    = {pos}")
print(f"  입력 방향    = {ori} (회전 없음)")
print(f"  DQ 실수부 q_r = {np.round(dq[:4], 4)}  ← 회전 정보")
print(f"  DQ 듀얼부 q_d = {np.round(dq[4:], 4)}  ← 위치 정보")

# %% [markdown]
"""
## 4. 듀얼 쿼터니언에서 위치 복원하기

DQ에 저장된 위치를 다시 꺼내는 공식입니다:

    position = 2 * q_d * conj(q_r) 의 벡터 부분

이 과정을 통해 DQ가 위치를 제대로 담고 있는지 검증할 수 있습니다.
"""

# %%
def quat_conj(q):
    """쿼터니언 켤레: [w, x, y, z] → [w, -x, -y, -z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def dq_to_position(dq):
    """듀얼 쿼터니언에서 위치 벡터 추출"""
    q_r = dq[:4]
    q_d = dq[4:]
    # 2 * q_d * conj(q_r) → 벡터 부분(x, y, z)이 위치
    t_quat = 2.0 * quat_mul(q_d, quat_conj(q_r))
    return t_quat[1:]  # w는 버리고 [x, y, z]만 반환

def dq_to_rotation(dq):
    """듀얼 쿼터니언에서 오일러 각도 추출"""
    q_r = dq[:4]
    # [w, x, y, z] → scipy [x, y, z, w]
    q_scipy = np.array([q_r[1], q_r[2], q_r[3], q_r[0]])
    return R.from_quat(q_scipy).as_euler('xyz')


# 검증: DQ → 다시 위치/방향으로 복원
pos_recovered = dq_to_position(dq)
ori_recovered = dq_to_rotation(dq)

print("DQ → 위치/방향 복원 검증")
print(f"  원래 위치 : {pos}")
print(f"  복원 위치 : {np.round(pos_recovered, 4)}")
print(f"  원래 방향 : {ori}")
print(f"  복원 방향 : {np.round(ori_recovered, 4)}")
print(f"  위치 오차 : {np.round(np.linalg.norm(pos - pos_recovered), 8)}")

# %% [markdown]
"""
## 5. 듀얼 쿼터니언 핵심 연산

### ① DQ 곱 (dq_mul)
두 DQ를 합성합니다. 개념적으로 **dq2가 dq1 끝에 붙어있는 구조**입니다.

```
world → [dq1] → frame1 → [dq2] → frame2
```

dq1 * dq2 는 world → frame2 의 변환입니다.

듀얼 파트(위치) 계산:
```
r1 * d2  →  dq2의 위치는 frame1 기준 → dq1의 회전(r1)으로 world 기준으로 변환
d1 * r2  →  dq1의 위치는 이미 world 기준 → 최종 회전계(r1*r2)에 맞게 정렬

합산     →  R1 @ t2  +  t1   (행렬 변환과 동일)
```

### ② DQ 켤레 (dq_conj)
DQ의 역변환입니다.
`dq * conj(dq) = 단위 DQ (아무것도 안 한 상태)`

단위 쿼터니언에서: `conj([q_r | q_d]) = [conj(q_r) | conj(q_d)]`
"""

# %%
def dq_mul(dq1, dq2):
    """
    두 DQ를 합성합니다. dq2가 dq1 끝에 붙어있는 구조.

    world → [dq1] → frame1 → [dq2] → frame2

    실수부: r1 * r2            (회전 합성)
    듀얼부: r1*d2 + d1*r2      (위치 합성)
      - r1*d2 : frame1 기준인 dq2 위치를 world 기준으로 변환
      - d1*r2 : world 기준인 dq1 위치를 최종 회전계에 정렬
    """
    r1, d1 = dq1[:4], dq1[4:]
    r2, d2 = dq2[:4], dq2[4:]
    return np.concatenate([
        quat_mul(r1, r2),
        quat_mul(r1, d2) + quat_mul(d1, r2),
    ])

def dq_conj(dq):
    """
    듀얼 쿼터니언 켤레 (역변환).
    [conj(q_r) | conj(q_d)]
    """
    return np.concatenate([quat_conj(dq[:4]), quat_conj(dq[4:])])


# 검증: dq * conj(dq) = 단위 DQ
dq_identity = dq_mul(dq, dq_conj(dq))

print("DQ 켤레 검증: dq * conj(dq) = 단위 DQ")
print(f"  결과 실수부 = {np.round(dq_identity[:4], 4)}  ← [1, 0, 0, 0] 이어야 함")
print(f"  결과 듀얼부 = {np.round(dq_identity[4:], 4)}  ← [0, 0, 0, 0] 이어야 함")

# %% [markdown]
"""
## 6. dq_rel 계산 (파지 순간의 상대 포즈)

두 팔이 물체를 쥔 직후, **왼팔(leader)과 오른팔(follower) 사이의 상대 관계**를 계산합니다.

```
dq_rel = conj(dq_leader) ⊗ dq_follower
```

이 공식의 의미:
- `conj(dq_leader)` = 왼팔 포즈의 역변환 (원점으로 되돌리기)
- `* dq_follower`   = 오른팔 포즈를 왼팔 기준으로 표현
- 결과 `dq_rel`     = "왼팔에서 오른팔까지의 상대적 변환"
"""

# %%
# 파지 순간 두 팔의 포즈 설정 (실제 시뮬레이터 값과 유사하게)
leader_pos = np.array([-0.15, 0.0, 0.4])    # 왼팔 EE 위치
leader_ori = np.array([0.0, np.pi/2, 0.0])  # 왼팔 EE 방향

follower_pos = np.array([0.15, 0.0, 0.4])   # 오른팔 EE 위치
follower_ori = np.array([0.0, np.pi/2, np.pi])  # 오른팔 EE 방향

# DQ로 변환
dq_leader   = pose_to_dq(leader_pos,   leader_ori)
dq_follower = pose_to_dq(follower_pos, follower_ori)

# 상대 포즈 계산
dq_rel = dq_mul(dq_conj(dq_leader), dq_follower)

print("=== 파지 순간 포즈 ===")
print(f"왼팔(leader)   위치: {leader_pos},  방향: {np.round(np.degrees(leader_ori), 1)}°")
print(f"오른팔(follower) 위치: {follower_pos}, 방향: {np.round(np.degrees(follower_ori), 1)}°")
print(f"\ndq_rel (8차원): {np.round(dq_rel, 4)}")
print(f"\ndq_rel이 표현하는 상대 위치: {np.round(dq_to_position(dq_rel), 4)}")
print(f"  → 오른팔은 왼팔 기준으로 x축으로 약 {np.round(dq_to_position(dq_rel)[0], 3)}m 떨어져 있음")

# %% [markdown]
"""
## 7. 강체 유지 검증

dq_rel을 이용해 오른팔 목표 포즈를 계산합니다.

```
dq_follower_target = dq_leader_new ⊗ dq_rel
```

왼팔이 어디로 이동하든 이 공식으로 오른팔의 위치가 자동 결정됩니다.
두 팔 사이 간격이 **항상 일정하게 유지** → 강체처럼 이동
"""

# %%
def compute_follower_pose(dq_leader_new, dq_rel):
    """
    왼팔 새 포즈와 dq_rel로 오른팔 목표 포즈 계산.
    반환: (position, euler_xyz)
    """
    dq_follower_new = dq_mul(dq_leader_new, dq_rel)
    pos = dq_to_position(dq_follower_new)
    ori = dq_to_rotation(dq_follower_new)
    return pos, ori


# 테스트: 왼팔을 z방향으로 0.1m 들어올리기
print("=== 강체 유지 검증 ===\n")
print("[이동 전]")
print(f"  왼팔  위치: {leader_pos}")
print(f"  오른팔 위치: {follower_pos}")
print(f"  두 팔 간격: {np.round(follower_pos - leader_pos, 4)}\n")

leader_new_pos = leader_pos + np.array([0, 0, 0.1])  # z+0.1 들어올리기
dq_leader_new  = pose_to_dq(leader_new_pos, leader_ori)
follower_new_pos, follower_new_ori = compute_follower_pose(dq_leader_new, dq_rel)

print("[이동 후 — z +0.1m 들어올리기]")
print(f"  왼팔  위치: {np.round(leader_new_pos, 4)}")
print(f"  오른팔 위치: {np.round(follower_new_pos, 4)}")
print(f"  두 팔 간격: {np.round(follower_new_pos - leader_new_pos, 4)}")
print(f"\n  ✓ 간격 변화: {np.round(np.linalg.norm((follower_new_pos - leader_new_pos) - (follower_pos - leader_pos)), 8)}")
print(f"    (0에 가까울수록 강체 유지 성공)")

# %% [markdown]
"""
## 8. 3D 시각화 — 원 궤적으로 강체 이동 확인

왼팔(leader)이 YZ 평면에서 원을 그릴 때,
오른팔(follower)이 어떻게 따라오는지 시각화합니다.

두 팔의 경로가 **동일한 형태의 원**을 그리면 강체 유지 성공입니다.
"""

# %%
# 원 궤적 생성
steps  = 32
radius = 0.05

leader_traj   = []
follower_traj = []

for i in range(steps + 1):
    angle = 2 * np.pi * i / steps

    # 왼팔: YZ 평면 원 이동
    lp = leader_pos + np.array([0,
                                 radius * np.cos(angle),
                                 radius * np.sin(angle)])
    dq_l = pose_to_dq(lp, leader_ori)
    fp, _ = compute_follower_pose(dq_l, dq_rel)

    leader_traj.append(lp)
    follower_traj.append(fp)

leader_traj   = np.array(leader_traj)
follower_traj = np.array(follower_traj)

# 시각화
fig = plt.figure(figsize=(12, 5))

# --- 3D 궤적 ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(leader_traj[:, 0],   leader_traj[:, 1],   leader_traj[:, 2],
         'b-o', markersize=3, label='Left arm (leader)')
ax1.plot(follower_traj[:, 0], follower_traj[:, 1], follower_traj[:, 2],
         'r-o', markersize=3, label='Right arm (follower)')

# 시작점 표시
ax1.scatter(*leader_traj[0],   color='blue',  s=80, marker='*', zorder=5)
ax1.scatter(*follower_traj[0], color='red',   s=80, marker='*', zorder=5)

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D Trajectory — Circle in YZ Plane')
ax1.legend()

# --- 두 팔 간격 변화 ---
ax2 = fig.add_subplot(122)
distances = np.linalg.norm(follower_traj - leader_traj, axis=1)
initial_dist = np.linalg.norm(follower_pos - leader_pos)

ax2.plot(distances, 'g-', linewidth=2, label='Arm distance')
ax2.axhline(y=initial_dist, color='gray', linestyle='--', label=f'Initial distance ({initial_dist:.4f}m)')
ax2.set_xlabel('Step')
ax2.set_ylabel('Distance between arms (m)')
ax2.set_title('Rigid Body Check — Distance Change')
ax2.legend()
ax2.set_ylim([initial_dist - 0.01, initial_dist + 0.01])

plt.tight_layout()
plt.savefig('dq_rigid_body_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: dq_rigid_body_visualization.png")

# %% [markdown]
"""
## 9. 정리

| 연산 | 공식 | 역할 |
|------|------|------|
| pose → DQ | `q_d = 0.5 * t ⊗ q_r` | 위치+방향을 DQ로 인코딩 |
| 상대 포즈 | `dq_rel = conj(dq_L) ⊗ dq_F` | 파지 순간 한 번만 계산 |
| 강체 이동 | `dq_F_new = dq_L_new ⊗ dq_rel` | leader 이동 시마다 follower 계산 |

**핵심 아이디어**:
- `dq_rel`은 두 팔 사이의 **불변 관계**
- leader가 어디로 움직이든 이 관계를 DQ 곱으로 복원
- pos+quat 방식은 위치/방향을 따로 보간하므로 강체 보장이 어렵지만,
  DQ 합성은 **회전과 이동을 동시에** 처리해 자연스러운 강체 이동을 보장
"""

# %%
print("=== Tutorial Complete ===")
print(f"dq_rel = {np.round(dq_rel, 4)}")
print("\nKey formula: dq_follower = dq_leader ⊗ dq_rel")
