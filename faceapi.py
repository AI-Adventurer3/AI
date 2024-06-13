from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FastAPI()

# 얼굴 분석기 초기화
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# 등록된 위험 인물 이미지와 얼굴 특징 저장 (여기서는 임시적으로 리스트에 저장)
registered_faces = []

@app.post("/register-dangerous-person/")
async def register_dangerous_person(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = face_app.get(img)

    if len(faces) > 0:
        registered_faces.append((file.filename, faces[0].normed_embedding))
        return JSONResponse(content={"message": "Dangerous person registered successfully."})
    else:
        return JSONResponse(content={"message": "No face detected in the image."}, status_code=400)

@app.post("/compare-face/")
async def compare_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = face_app.get(img)

    if len(faces) == 0:
        return JSONResponse(content={"message": "No face detected in the uploaded image."}, status_code=400)

    feat1 = np.array(faces[0].normed_embedding, dtype=np.float32)
    results = []

    for ref_filename, ref_embedding in registered_faces:
        feat2 = np.array(ref_embedding, dtype=np.float32)
        similarity = np.dot(feat1, feat2.T)

        # 유사도 임계값 (임계값은 필요에 따라 조정 가능)
        threshold = 0.5

        if similarity > threshold:
            results.append(ref_filename)

    if results:
        return JSONResponse(content={"match": True, "matched_files": results})
    else:
        return JSONResponse(content={"match": False})
