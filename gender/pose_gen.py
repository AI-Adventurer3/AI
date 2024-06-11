# import cv2
# import io
# import numpy as np
# from fastapi import FastAPI, UploadFile, File
# from transformers import pipeline

# # Initialize the classifier
# classifier = pipeline("image-classification", model="dima806/man_woman_face_image_detection")

# app = FastAPI()

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded file
#         byte_file = await file.read()

#         # Convert byte array to binary stream
#         image_bin = io.BytesIO(byte_file)

#         # Try to open the image using PIL
#         try:
#             pil_img = PIL.Image.open(image_bin)
#         except PIL.UnidentifiedImageError:
#             return {"error": "Cannot identify image file. Please upload a valid image."}

#         # Classification
#         result = classifier(pil_img)  # Use the PIL image directly
        
#         return {"result": result}
#     except Exception as e:
#         return {"error": str(e)}

# @app.get("/webcam/")
# async def webcam_classification():
#     try:
#         # Open webcam
#         cap = cv2.VideoCapture(0)

#         # Check if the webcam is opened successfully
#         if not cap.isOpened():
#             return {"error": "Unable to open webcam."}

#         # Read frame from webcam
#         ret, frame = cap.read()

#         # Convert frame to PIL Image
#         pil_img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         # Classification
#         result = classifier(pil_img)  # Use the PIL image directly

#         # Release the webcam
#         cap.release()

#         return {"result": result}
#     except Exception as e:
#         return {"error": str(e)}


# 웹으로 성별구분 코드
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

# 감정 분류 파이프라인 설정
classifier = pipeline("image-classification", model="dima806/man_woman_face_image_detection")

def detect_emotion(frame):
    # 프레임을 PIL 이미지로 변환
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 이미지를 분류
    result = classifier(pil_img)
    
    return result

def main():
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 감정 탐지
        result = detect_emotion(frame)
        
        # 감정 결과를 프레임에 표시
        y_offset = 30
        for res in result:
            label = res['label']
            score = res['score']
            text = f'{label}: {score:.2f}'
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30

        # 프레임을 화면에 표시
        cv2.imshow("Webcam Emotion Detection", frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
