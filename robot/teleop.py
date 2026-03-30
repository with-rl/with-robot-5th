"""키보드 텔레오퍼레이션 — WASD + 방향키로 로봇 이동."""
import time
import threading
import numpy as np
import requests
from pynput import keyboard

BASE_URL = "http://localhost:8800"
STEP_M   = 0.3              # 한 스텝 이동 거리 (m)
STEP_RAD = np.deg2rad(15)   # 한 스텝 회전 각도
CONTROL_HZ = 10             # 제어 루프 주기 (Hz)

pressed  = set()
_lock    = threading.Lock()
_running = True


def get_pose():
    m = requests.get(f"{BASE_URL}/state", timeout=1.0).json()["mobile"]
    return m["x"], m["y"], m["theta"]


def send_target(x, y, theta):
    """목표 위치만 설정하고 즉시 반환 (timeout=0.05 → fire-and-forget)."""
    code = f"set_mobile_target_position([{x:.4f}, {y:.4f}, {theta:.4f}], timeout=0.05)"
    requests.post(f"{BASE_URL}/send_action", json={
        "action": {"type": "run_code", "payload": {"code": code}}
    }, timeout=2.0)


def on_press(key):
    with _lock:
        pressed.add(key)


def on_release(key):
    with _lock:
        pressed.discard(key)
    if key == keyboard.KeyCode.from_char('q'):
        global _running
        _running = False
        return False  # Listener 종료


def control_loop():
    print("\n[Teleop] 시작")
    print("  ↑ / ↓  : 전진 / 후진")
    print("  ← / →  : 좌회전 / 우회전")
    print("  Q       : 종료\n")

    UP    = keyboard.Key.up
    DOWN  = keyboard.Key.down
    LEFT  = keyboard.Key.left
    RIGHT = keyboard.Key.right

    while _running:
        with _lock:
            keys = frozenset(pressed)

        if keys:
            try:
                x, y, theta = get_pose()
                dx = dy = dth = 0.0

                if UP in keys:
                    dx += np.cos(theta) * STEP_M
                    dy += np.sin(theta) * STEP_M
                if DOWN in keys:
                    dx -= np.cos(theta) * STEP_M
                    dy -= np.sin(theta) * STEP_M
                if LEFT in keys:
                    dth += STEP_RAD
                if RIGHT in keys:
                    dth -= STEP_RAD

                if dx or dy or dth:
                    tx, ty, tth = x + dx, y + dy, theta + dth
                    send_target(tx, ty, tth)
                    print(f"\r→ target ({tx:.2f}, {ty:.2f}, {np.rad2deg(tth):.1f}°)    ", end="")

            except Exception as e:
                print(f"\n[오류] {e}")

        time.sleep(1.0 / CONTROL_HZ)


if __name__ == "__main__":
    t = threading.Thread(target=control_loop, daemon=True)
    t.start()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    _running = False
    print("\n[Teleop] 종료")
