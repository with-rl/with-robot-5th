import sys
import cv2
import requests
import base64
import numpy as np
import time
import matplotlib.pyplot as plt
import open3d as o3d

BASE_URL = "http://localhost:8800"

# ============================================================
# Camera intrinsics (robot0_head_camera)
#   fovy=75°, resolution=640×480, square pixels
# ============================================================
CAM_W, CAM_H = 640, 480
_FOVY_RAD = np.deg2rad(75.0)
FY = (CAM_H / 2) / np.tan(_FOVY_RAD / 2)   # ≈ 312.7
FX = FY
CX, CY = CAM_W / 2, CAM_H / 2

# Camera extrinsics: camera frame → mobilebase body frame
# From XML: pos="0.05 0 0.9", xyaxes="0 -1 0 0.5 0 0.866"
#   camera X-axis in body: ( 0,    -1,    0    )
#   camera Y-axis in body: ( 0.5,   0,    0.866)
#   camera Z-axis in body: ( X × Y ) = (-0.866, 0, 0.5)
_R = np.array([
    [ 0,    -0.5,   0.866],
    [-1,     0,     0    ],
    [ 0,    -0.866, -0.5 ],
], dtype=np.float64)
_t = np.array([0.05, 0.0, 0.9], dtype=np.float64)

T_BODY_FROM_CAM = np.eye(4)
T_BODY_FROM_CAM[:3, :3] = _R
T_BODY_FROM_CAM[:3, 3] = _t

# Depth 유효 범위
DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 5.0

GRID_SIZE = 0.1  # voxel 크기 (m)


# ============================================================
# 헬퍼 함수
# ============================================================

def get_robot_pose() -> np.ndarray:
    """GET /state → [x, y, theta] (world frame)."""
    data = requests.get(f"{BASE_URL}/state", timeout=2.0).json()
    m = data["mobile"]
    return np.array([m["x"], m["y"], m["theta"]])


def depth_to_pointcloud(depth_mm: np.ndarray) -> np.ndarray:
    """
    depth_mm (H, W) uint16 → 카메라 좌표계 3D 포인트 (N, 3).
    유효 범위 밖 픽셀은 제외.
    """
    depth_m = depth_mm.astype(np.float32) / 1000.0
    mask = (depth_m > DEPTH_MIN_M) & (depth_m < DEPTH_MAX_M)
    v, u = np.where(mask)
    z = depth_m[v, u]
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return np.stack([x, y, z], axis=1)  # (N, 3)


def transform_to_world(pts_cam: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    """
    카메라 좌표계 (N, 3) → 월드 좌표계 (N, 3).
    robot_pose: [x, y, theta]  (mobile base, z=0 가정)
    """
    rx, ry, theta = robot_pose
    c, s = np.cos(theta), np.sin(theta)

    T_world_body = np.array([
        [c, -s, 0, rx],
        [s,  c, 0, ry],
        [0,  0, 1,  0],
        [0,  0, 0,  1],
    ], dtype=np.float64)

    T_world_cam = T_world_body @ T_BODY_FROM_CAM

    ones = np.ones((len(pts_cam), 1))
    pts_h = np.hstack([pts_cam, ones])          # (N, 4)
    pts_world = (T_world_cam @ pts_h.T).T       # (N, 4)
    return pts_world[:, :3]


# ============================================================
# GridMap3D
# ============================================================

class GridMap3D:
    """Depth 프레임을 누적해 3D 점유 voxel map을 구성."""

    def __init__(self, grid_size: float = GRID_SIZE):
        self.grid_size = grid_size
        self.voxels: dict = {}  # (ix, iy, iz) → hit count

    def update(self, depth_mm: np.ndarray, robot_pose: np.ndarray) -> int:
        """새 depth 프레임으로 map 갱신. 추가된 포인트 수 반환."""
        pts_cam = depth_to_pointcloud(depth_mm)
        if len(pts_cam) == 0:
            return 0
        pts_world = transform_to_world(pts_cam, robot_pose)
        indices = np.floor(pts_world / self.grid_size).astype(int)
        # numpy로 중복 집계하여 Python loop 제거
        unique_idx, counts = np.unique(indices, axis=0, return_counts=True)
        for idx, cnt in zip(map(tuple, unique_idx), counts):
            self.voxels[idx] = self.voxels.get(idx, 0) + int(cnt)
        return len(pts_cam)

    def save(self, path: str = "grid_3d_map.npy"):
        np.save(path, {"voxels": self.voxels, "grid_size": self.grid_size}, allow_pickle=True)
        print(f"[GridMap3D] 저장 완료: {path}  (voxels={len(self.voxels)})")

    @classmethod
    def load(cls, path: str = "grid_3d_map.npy") -> "GridMap3D":
        data = np.load(path, allow_pickle=True).item()
        obj = cls(grid_size=data["grid_size"])
        obj.voxels = data["voxels"]
        print(f"[GridMap3D] 로드 완료: {path}  (voxels={len(obj.voxels)})")
        return obj

    def stats(self):
        total = len(self.voxels)
        hits = sum(self.voxels.values())
        print(f"[GridMap3D] voxels={total}, total_hits={hits}, grid_size={self.grid_size}m")

    def visualize_topdown(self, min_hits: int = 2):
        """XY 평면 top-down 2D occupancy 시각화."""
        occupied = [k for k, v in self.voxels.items() if v >= min_hits]
        if not occupied:
            print("[GridMap3D] 표시할 voxel 없음")
            return
        xs = [k[0] * self.grid_size for k in occupied]
        ys = [k[1] * self.grid_size for k in occupied]
        plt.figure(figsize=(10, 10))
        plt.scatter(xs, ys, s=1, c="black")
        plt.axis("equal")
        plt.title(f"Top-down occupancy map  (grid={self.grid_size}m, voxels={len(occupied)})")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.tight_layout()
        plt.show()

    def visualize_3d(self, min_hits: int = 2, max_points: int = 50_000):
        """Open3D VoxelGrid 시각화 (0.1m 큐브)."""
        occupied = [k for k, v in self.voxels.items() if v >= min_hits]
        if not occupied:
            print("[GridMap3D] 표시할 voxel 없음")
            return
        if len(occupied) > max_points:
            occupied = occupied[:max_points]
        centers = np.array(occupied, dtype=np.float64) * self.grid_size
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers)
        z_vals = centers[:, 2]
        t = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-9)
        pcd.colors = o3d.utility.Vector3dVector(plt.cm.viridis(t)[:, :3])
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.grid_size)
        o3d.visualization.draw_geometries(
            [voxel_grid],
            window_name=f"3D Grid Map ({len(occupied)} voxels)",
            width=1024, height=768,
        )


# ============================================================
# 수집 루프
# ============================================================

def build_grid_map(camera_name="robot0_head_camera", interval=0.5):
    """
    로봇이 이동하는 동안 depth를 수집해 3D grid map을 구성.
    Ctrl+C로 종료하면 저장 후 VoxelGrid로 시각화.
    """
    url = f"{BASE_URL}/camera?camera={camera_name}"
    grid_map = GridMap3D()
    frame_count = 0

    print("[GridMap3D] 수집 시작. Ctrl+C로 종료합니다.")

    try:
        while True:
            try:
                robot_pose = get_robot_pose()

                data = requests.get(url, timeout=2.0).json()
                if data.get("status") != "success":
                    print(f"\nAPI 에러: {data.get('message')}")
                    time.sleep(1)
                    continue

                depth_bytes = base64.b64decode(data["depth_base64"])
                depth_mm = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                added = grid_map.update(depth_mm, robot_pose)
                frame_count += 1

                x, y, th = robot_pose
                print(f"\r[frame={frame_count}] pose=({x:.2f}, {y:.2f}, {np.rad2deg(th):.1f}°)"
                      f"  +{added} pts  voxels={len(grid_map.voxels)}", end="")

                time.sleep(interval)

            except requests.exceptions.ConnectionError:
                print("\n연결 실패. 서버 재시도 대기 중...")
                time.sleep(2)
            except Exception as e:
                print(f"\n오류: {e}")
                break

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 수집 중단.")

    print()
    grid_map.stats()
    grid_map.save("grid_3d_map.npy")
    grid_map.visualize_3d()
    return grid_map


def show_grid_map(path: str = "grid_3d_map.npy"):
    """저장된 grid_3d_map.npy를 불러와 3D 시각화."""
    grid_map = GridMap3D.load(path)
    grid_map.stats()
    grid_map.visualize_3d()



# ============================================================
# 기존 모니터링 함수
# ============================================================

def monitor_camera(camera_name="robot0_head_camera"):
    url = f"http://localhost:8800/camera?camera={camera_name}"
    print(f"[{camera_name}] 실시간 모니터링을 시작합니다.")
    print("종료하려면 화면을 클릭하고 'q'키를 누르세요.")

    while True:
        try:
            start_time = time.time()
            
            # API 요청으로 데이터 받아오기
            response = requests.get(url, timeout=2.0)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "success":
                # Base64 문자열에서 이미지 디코딩
                rgb_bytes = base64.b64decode(data["rgb_base64"])
                np_img = np.frombuffer(rgb_bytes, np.uint8)
                frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                # 프레임 레이트 (FPS) 계산
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Base64 문자열에서 Depth 이미지 디코딩
                depth_bytes = base64.b64decode(data["depth_base64"])
                np_depth = np.frombuffer(depth_bytes, np.uint8)
                depth_mm = cv2.imdecode(np_depth, cv2.IMREAD_UNCHANGED)
                
                # 화면 표출용으로 Depth 맵 정규화 및 색상(히트맵) 입히기
                # (0 ~ 3000mm 범위 정도를 기준으로 잘라내어 시각적 대비를 높임)
                depth_clipped = np.clip(depth_mm, 0, 3000)
                depth_visual = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(255 - depth_visual, cv2.COLORMAP_JET)
                
                cv2.putText(depth_colormap, "Depth Map", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # RGB 이미지와 Depth 이미지를 가로로 이어 붙이기
                combined_frame = np.hstack((frame, depth_colormap))

                # 화면에 송출
                cv2.imshow("Robot Live RGB-D Monitor", combined_frame)

                # 'q' 키를 누르면 루프 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n모니터링을 종료합니다.")
                    break
            else:
                print(f"\nAPI 에러: {data.get('message')}")
                time.sleep(1)

        except requests.exceptions.Timeout:
            print("\n요청 시간 초과. 서버가 응답하지 않습니다.")
        except requests.exceptions.ConnectionError:
            print("\n연결 실패. 서버가 꺼져있는지 확인하세요. (main.py 재시도 대기중...)")
            time.sleep(2)
        except Exception as e:
            print(f"\n오류 발생: {e}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # python camera_client.py         → RGB-D 실시간 모니터링
    # python camera_client.py map     → 3D grid map 수집 (Ctrl+C 시 자동 저장)
    # python camera_client.py show    → 저장된 grid_3d_map.npy 시각화
    if len(sys.argv) > 1 and sys.argv[1] == "map":
        build_grid_map()
    elif len(sys.argv) > 1 and sys.argv[1] == "show":
        path = sys.argv[2] if len(sys.argv) > 2 else "grid_3d_map.npy"
        show_grid_map(path)
    else:
        monitor_camera()
