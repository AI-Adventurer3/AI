from fastapi import FastAPI, File, BackgroundTasks, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import openai
import re
import os
import random
import string
import base64
import shutil
import io
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import shutil
import tempfile
import traceback
import time

# OpenAI API 키 설정!
openai.api_key = ''

# 모델 초기화
obj_detector = pipeline("object-detection", model="nickmuchi/yolos-base-finetuned-masks")
face_classifier = pipeline("image-classification", model="AliGhiasvand86/gisha_coverd_uncoverd_face")
face_expression = pipeline("image-classification", model="trpakov/vit-face-expression")
image_captioning = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", framework="pt")
age_classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# 얼굴 분석기 설정
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))  # 얼굴 분석기 준비

def generate_random_filename():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))

def delete_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"파일 삭제: {file_path}")
        else:
            print(f"파일이 존재하지 않습니다: {file_path}")
    except Exception as e:
        print(f"파일 삭제 중 오류 발생: {str(e)}")

async def save_upload_file(upload_file: UploadFile) -> str:
    try:
        contents = await upload_file.read()
        random_filename = generate_random_filename()
        save_path = os.path.join(UPLOAD_DIRECTORY, f"{random_filename}_{upload_file.filename}")
        with open(save_path, "wb") as f:
            f.write(contents)
        return save_path
    except Exception as e:
        raise e

def create_temp_file(original_path: str) -> str:
    """한글 파일 이름 문제를 피하기 위해 임시 파일을 생성합니다."""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")  # PNG 확장자로 임시 파일 생성
        shutil.copyfile(original_path, temp_file.name)
        return temp_file.name
    except Exception as e:
        raise ValueError(f"임시 파일을 생성하는 중 오류 발생: {str(e)}")

def compare_faces(image_path1: str, image_path2: str) -> float:
    """두 이미지 파일 경로를 받아 얼굴 임베딩을 비교하고 유사도를 반환합니다."""
    try:
        # 경로 디버깅 정보 출력
        print(f"비교할 첫 번째 이미지 경로: {image_path1}")
        print(f"비교할 두 번째 이미지 경로: {image_path2}")

        # 임시 파일 생성
        temp_image_path1 = create_temp_file(image_path1)
        temp_image_path2 = create_temp_file(image_path2)

        img1 = cv2.imread(temp_image_path1)
        img2 = cv2.imread(temp_image_path2)

        if img1 is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {temp_image_path1}")
        if img2 is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {temp_image_path2}")

        faces1 = face_app.get(img1)
        faces2 = face_app.get(img2)

        if len(faces1) == 0 or len(faces2) == 0:
            raise ValueError("한 이미지 또는 두 이미지에서 얼굴을 찾을 수 없습니다.")

        embedding1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
        embedding2 = np.array(faces2[0].normed_embedding, dtype=np.float32)

        # 코사인 유사도 계산
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

        # 임시 파일 삭제
        os.remove(temp_image_path1)
        os.remove(temp_image_path2)

        return similarity_score
    except Exception as e:
        # 에러 로그 추가
        print(f"얼굴 비교 중 오류 발생: {str(e)}")
        traceback.print_exc()
        raise e
    
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

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        save_path = await save_upload_file(file)
        delete_file(save_path)
        return JSONResponse(status_code=200, content={"message": "Image uploaded successfully", "file_path": save_path})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes))

        obj_results = obj_detector(image)
        face_results = face_classifier(image)
        face_exp_results = face_expression(image)
        age_results = age_classifier(image)
        caption_results = image_captioning(image)

        # 이미지를 Base64로 인코딩하여 결과 객체에 추가
        image_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # 얼굴 가림 여부 확인
        face_status = "uncovered" if face_results and face_results[0]['score'] > 0.6 else "covered"

        # 표정 인식 결과에서 조건에 맞는 상위 두 개의 감정 레이블 추출
        expression_result_sorted = sorted(face_exp_results, key=lambda x: x['score'], reverse=True)
        top_two_expressions = []
        if expression_result_sorted and expression_result_sorted[0]['score'] > 0.6:
            top_two_expressions.append(expression_result_sorted[0]['label'])
        if len(expression_result_sorted) > 1 and expression_result_sorted[1]['score'] > 0.4:
            top_two_expressions.append(expression_result_sorted[1]['label'])

        # 나이 예측 결과에서 두 번째로 높은 값 추출
        age_result_sorted = sorted(age_results, key=lambda x: x['score'], reverse=True)
        second_highest_age_label = age_result_sorted[1]['label'] if len(age_result_sorted) > 1 else age_result_sorted[0]['label']

        # 얼굴 인식 및 동일 인물 여부 판단
        faces = face_app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        if len(faces) == 0:
            return JSONResponse(status_code=400, content={"error": "이미지에서 얼굴을 찾을 수 없습니다."})

        # 얼굴 유사도 계산
        # 비교를 위해 다른 이미지(예: 정해진 기준 이미지)와의 유사도를 비교할 수 있도록 확장 가능
        comparison_image_path = "path/to/your/reference/image.png"  # 기준 이미지 경로 설정
        if os.path.exists(comparison_image_path):
            similarity_score = compare_faces(comparison_image_path, file.filename)  # 업로드된 파일과 비교
        else:
            similarity_score = None

        # 간단한 요약 생성
        summary = (
            f"얼굴을 {face_status}한 "
            f"이 {', '.join(top_two_expressions)} 표정으로 "
            f"나이는 {second_highest_age_label} "
            f"{caption_results[0]['generated_text']}"
        )

        # OpenAI GPT 모델을 사용하여 요약 문장을 자연스럽게 만들기
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Make it short and sound natural and change it to Korean without [] {summary}"}
            ],
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        # 요약된 문장 출력
        final_summary = response['choices'][0]['message']['content'].strip()

        # 폭력적이거나 위험한 문장인지 확인
        def is_violent_or_dangerous(sentence):
            # 폭력적인 단어 리스트
            violent_keywords = ['주먹', '발길질', '때리다', '공격', '싸움', '폭력', '협박', '위협', '칼', '총', '무기', '당근']
            # 위험한 감정
            emotion_keywords = ['슬픈', '화난', '두려운']
            # 위험한 상황 단어 리스트
            dangerous_keywords = ['불', '화재', '추락', '붕괴', '폭발', '전기', '피', '상처', '응급', '구조', '부상']

            # 정규 표현식을 사용하여 문장에서 폭력적이거나 위험한 단어가 있는지 확인
            for keyword in violent_keywords + emotion_keywords + dangerous_keywords:
                if re.search(keyword, sentence):
                    return True
            return False

        # 최종 요약 문장이 폭력적이거나 위험한지 확인
        is_dangerous = is_violent_or_dangerous(final_summary)

        result = {
            "face_status": face_status,
            "expressions": top_two_expressions,
            "caption": caption_results[0]['generated_text'],
            "age": second_highest_age_label,
            "summary": final_summary,
            "is_dangerous": is_dangerous,
            "image_base64": image_base64,
            "similarity_score": similarity_score  # 얼굴 유사도 점수 추가
        }

        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/compare-faces/")
async def compare_faces_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        # 업로드된 파일 저장
        file1_path = await save_upload_file(file1)
        file2_path = await save_upload_file(file2)

        # 이미지 비교 함수 호출
        similarity_score = compare_faces(file1_path, file2_path)

        # 비교 결과 반환7
        return JSONResponse(content={"similarity_score": float(similarity_score)})  # float으로 변환하여 JSON 직렬화 문제 해결
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
