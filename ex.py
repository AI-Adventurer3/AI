from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
import io

# 이미지 분류 파이프라인 설정
classifier = pipeline("image-classification", model="trpakov/vit-face-expression")

app = FastAPI()

@app.post("/face_expression/")
async def create_upload_file(file: UploadFile):
    # 업로드된 파일 읽기
    byte_file = await file.read()
    
    # 이미지를 바이너리 스트림으로 변환
    image_bin = io.BytesIO(byte_file)
    
    # PIL 이미지로 변환
    pil_img = Image.open(image_bin)
    
    # 이미지를 분류
    result = classifier(pil_img)
    
    # 결과 출력 및 반환
    print(result)
    return JSONResponse(content=result)

