from fastapi import FastAPI, BackgroundTasks
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import threading
import time

app = FastAPI()

# Step 2 추론기 만듬
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640)) # prepare 얼굴분석기

# 전역 변수로 종료 플래그 추가
exit_flag = False

# Step 3 웹캠 연결 및 캡쳐
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

# 종료 신호를 받기 위한 함수
def wait_for_exit():
    global exit_flag
    input("종료하려면 Enter 키를 누르세요...\n")
    exit_flag = True

# Step 4 여러 기준 이미지 가져오기
reference_images = ['jm.png', 'yr.jpg', 'mj.jpg']
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

# 종료 신호를 대기하는 스레드 시작
exit_thread = threading.Thread(target=wait_for_exit)
exit_thread.start()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.post("/start-capture/")
async def start_capture(background_tasks: BackgroundTasks):
    global exit_flag
    exit_flag = False
    background_tasks.add_task(capture_face_from_webcam)
    return {"message": "Capture started"}

@app.post("/stop-capture/")
async def stop_capture():
    global exit_flag
    exit_flag = True
    return {"message": "Capture stopped"}

# Step 5 캡처된 이미지 처리
@app.post("/process-images/")
async def process_images():
    capture_index = 1
    results = []
    while True:
        img_path = f"webcam_capture{capture_index}.jpg"
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            break

        faces1 = face_app.get(img)
        if len(faces1) == 0:
            print(f"얼굴 검출 실패: {img_path}")
            capture_index += 1
            continue

        # Step 6 후처리 출력
        rimg = face_app.draw_on(img, faces1) # 이미지 위에 얼굴 검출 결과 그리기
        output_annotated_path = f"./webcam_capture{capture_index}_annotated.jpg"
        cv2.imwrite(output_annotated_path, rimg) # 결과 이미지를 파일로 저장

        print(len(faces1))
        print(faces1[0].embedding)

        # then print all-to-all face similarity
        feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32) # normed_embedding: 얼굴의 고유한 특징을 나타내는 벡터

        for ref_path, ref_embedding in reference_faces:
            feat2 = np.array(ref_embedding, dtype=np.float32)
            sims = np.dot(feat1, feat2.T) # np.dot: 두 벡터의 내적을 계산하여 유사도를 측정
            print(f"유사도 ({ref_path}): {sims}")

            # 얼굴 유사도 판단 및 출력
            threshold = 0.5 # 유사도 임계값 설정 (0.5는 예시 값이며 조정 가능)

            if sims > threshold:
                result = f"{img_path}와 {ref_path}: 동일 인물 입니다."
            else:
                result = f"{img_path}와 {ref_path}: 다른 사람 입니다."
            results.append(result)

        # 캡쳐 인덱스 증가
        capture_index += 1

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
