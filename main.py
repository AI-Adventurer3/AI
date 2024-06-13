from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import os
import random
import string
import base64

app = FastAPI()

UPLOAD_DIRECTORY = "uploads"

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

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        save_path = await save_upload_file(file)
        return JSONResponse(status_code=200, content={"message": "Image uploaded successfully", "file_path": save_path})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from transformers import pipeline

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

obj_detector = pipeline("object-detection", model="nickmuchi/yolos-base-finetuned-masks")
face_classifier = pipeline("image-classification", model="AliGhiasvand86/gisha_coverd_uncoverd_face")
face_expression = pipeline("image-classification", model="trpakov/vit-face-expression")
gender_detector = pipeline("image-classification", model="dima806/man_woman_face_image_detection")
image_captioning = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", framework="pt")

@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes))

        obj_results = obj_detector(image)
        face_results = face_classifier(image)
        face_exp_results = face_expression(image)
        gender_results = gender_detector(image)
        caption_results = image_captioning(image)

        # 이미지를 Base64로 인코딩하여 결과 객체에 추가
        image_base64 = base64.b64encode(img_bytes).decode('utf-8')
        results = {
            "captions": [result['generated_text'] for result in caption_results],
            "emotions": [{"label": result['label'], "score": result['score']} for result in face_exp_results],
            "genders": [{"label": result['label'], "score": result['score']} for result in gender_results],
            "faces": [{"label": result['label'], "score": result['score']} for result in face_results],
            "objects": [{"label": result['label'], "score": result['score']} for result in obj_results],
            "image_base64": image_base64
        }

        return results
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
