import cv2
import requests
import base64
import numpy as np
import time

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
    monitor_camera()
