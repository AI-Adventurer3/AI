from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import base64
from transformers import pipeline

app = FastAPI()

# 기존 모델들
obj_detector = pipeline("object-detection", model="nickmuchi/yolos-base-finetuned-masks")
face_classifier = pipeline("image-classification", model="AliGhiasvand86/gisha_coverd_uncoverd_face")
face_expression = pipeline("image-classification", model="trpakov/vit-face-expression")
gender_detector = pipeline("image-classification", model="dima806/man_woman_face_image_detection")
image_captioning = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", framework="pt")


# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result_html": ""})

@app.post("/classify-image/", response_class=HTMLResponse)
async def classify_image(request: Request, file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes))

    emotion_translations = {
        'fear': '공포',
        'sad': '슬픔',
        'angry': '화남',
        'neutral': '평화',
        'happy': '행복'
    }

    gender_translations = {
        'man': '남자',
        'woman': '여자'
    }

    # 객체 감지 결과
    obj_results = obj_detector(image)
    # 얼굴 커버 분류 결과
    face_results = face_classifier(image)
    # 얼굴 표정 감지 결과
    face_exp_results = face_expression(image)
    # 성별 감지 결과
    gender_results = gender_detector(image)
    # 이미지 캡셔닝 결과
    caption_results = image_captioning(image)

    result_html = "<h2>업로드 이미지:</h2>"
    result_html += f'<img src="data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"/>'

    combined_results = []

    for result in caption_results:
        combined_results.append(result['generated_text'])

    for result in face_exp_results:
        translated_label = emotion_translations.get(result['label'], result['label'])
        combined_results.append(f"{translated_label}: {result['score']:.2f}")

    for result in gender_results:
        translated_label = gender_translations.get(result['label'], result['label'])
        combined_results.append(f"{translated_label}: {result['score']:.2f}")

    for result in face_results:
        if result['label'] == 'uncovered' and result['score'] > 0.51:
            combined_results.append(f"얼굴 공개: {result['score']:.2f}")
        elif result['label'] == 'uncovered' and result['score'] <= 0.51:
            combined_results.append(f"얼굴 비공개: {result['score']:.2f}")
        elif result['label'] == 'covered':
            combined_results.append(f"얼굴 비공개: {result['score']:.2f}")

    for result in obj_results:
        score_percentage = f"{result['score'] * 100:.2f}"
        if result['label'] == 'with_mask':
            label = "마스크를 착용함"
        else:
            label = result['label']
        combined_results.append(f"{label}: {score_percentage} % 확률로 씀")
    if not obj_results:
        combined_results.append("마스크를 착용하지 않음")
        
    result_html += "<p>" + " ".join(combined_results) + "</p>"
    return templates.TemplateResponse("index.html", {"request": request, "result_html": result_html})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
