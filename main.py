from fastapi import FastAPI, File, UploadFile, Request
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
import io
from PIL import Image

# OpenAI API 키 설정
openai.api_key = ""

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

def generate_random_filename():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))

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

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        save_path = await save_upload_file(file)
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

        # 나이 예측 결과에서 두 번째로 높은 값 추출출
        age_result_sorted = sorted(age_results, key=lambda x: x['score'], reverse=True)
        second_highest_age_label = age_result_sorted[1]['label'] if len(age_result_sorted) > 1 else age_result_sorted[0]['label']

        # 간단한 요약 생성성
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

        # 폭력적이거나 위험한 문장인지 확인하는 함수
        def is_violent_or_dangerous(sentence):
            # 폭력적인 단어 리스트
            violent_keywords = ['주먹', '발길질', '때리다', '공격', '싸움', '폭력', '협박', '위협', '칼', '총', '무기', '당근']
            # 위험한 감정
            emotion_keywords = ['슬픈', '화난','두려운']
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
            "image_base64": image_base64
        }

        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
