from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 얼굴 분석기 설정
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))  # prepare 얼굴분석기

exit_flag = False

# 실시간 비디오 피드를 중지
@app.post("/stop_video_feed/")
async def stop_video_feed():
    global exit_flag
    exit_flag = True
    return {"message": "실시간 비디오 피드 중지"}

# 웹캠에서 얼굴 캡처 시작
@app.post("/start-capture/")
async def start_capture(background_tasks: BackgroundTasks):
    global exit_flag
    exit_flag = False
    background_tasks.add_task(capture_face_from_webcam)
    return {"message": "웹캠 얼굴 캡처 시작"}

# 웹캠 얼굴 캡처 중지
@app.post("/stop-capture/")
async def stop_capture():
    global exit_flag
    exit_flag = True
    return {"message": "Capture stopped"}

def capture_face_from_webcam(output_path_template="webcam_capture{}.jpg"):
    global exit_flag
    cap = cv2.VideoCapture(0)  # 0번 카메라를 엽니다.

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    capture_index = 1

    while not exit_flag:
        ret, frame = cap.read()
        if not ret:
            print("캡쳐 실패")
            break

        # 얼굴 검출
        faces = face_app.get(frame)

        # 얼굴이 검출되면 사각형으로 표시
        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # 얼굴이 검출된 프레임을 저장하고 반환
        if faces:
            output_path = output_path_template.format(capture_index)
            cv2.imwrite(output_path, frame)
            print(f"캡쳐 저장됨: {output_path}")
            capture_index += 1

        # 프레임을 화면에 표시
        cv2.imshow("Webcam Emotion, Captioning, and Gender Detection", frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 1초 마다 캡처
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

# 참고 이미지 로드
reference_images = ['jm.png', 'yr.jpg', 'mj.jpg',
                    'im01.PNG', 'im02.PNG', 'im03.PNG','im04.PNG', 'im05.PNG','im06.PNG',
                    'im07.PNG', 'im08.PNG', 'im09.PNG', 'im10.PNG', 'im11.PNG', 'im12.PNG',
                    'im13.PNG', 'im14.PNG', 'im15.PNG', 'im16.PNG']
reference_faces = []

for img_path in reference_images:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"{img_path}에서 이미지를 로드할 수 없습니다.")
        continue

    faces = face_app.get(img)
    if len(faces) == 0:
        print(f"{img_path}에서 얼굴을 검출하지 못했습니다.")
        continue

    reference_faces.append((img_path, faces[0].normed_embedding))

# 캡처된 이미지에서 얼굴을 검출
# 참고 이미지와 비교하여 동일 인물 여부 판단
@app.post("/process-images/")
async def process_images(request: Request):
    data = await request.json()
    captured_images = data.get('images', [])
    similarities = []

    for img_data in captured_images:
        img_str = base64.b64decode(img_data.split(",")[1])
        img = Image.open(BytesIO(img_str)).convert("RGB")
        img = np.array(img)

        faces1 = face_app.get(img)
        if len(faces1) == 0:
            print("얼굴 검출 실패")
            continue

        # 이미지 위에 얼굴 검출 결과 그리기
        rimg = face_app.draw_on(img, faces1)

        feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)

        for ref_path, ref_embedding in reference_faces:
            feat2 = np.array(ref_embedding, dtype=np.float32)
            sims = np.dot(feat1, feat2.T)
            print(f"유사도 ({ref_path}): {sims}")
            similarities.append(f"유사도 ({ref_path}): {sims}")

    return {"similarities": similarities}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
