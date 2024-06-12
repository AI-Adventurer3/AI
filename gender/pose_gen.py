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
# 123123123
# import cv2
# from transformers import pipeline
# from PIL import Image
# import numpy as np

# # 감정 분류 파이프라인 설정
# classifier = pipeline("image-classification", model="touchtech/fashion-images-gender-age-vit-large-patch16-224-in21k-v2")
# # 성별/연령대 모델 : touchtech/fashion-images-gender-age-vit-large-patch16-224-in21k-v2
# def detect_emotion(frame):
#     # 프레임을 PIL 이미지로 변환
#     pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # 이미지를 분류
#     result = classifier(pil_img)
    
#     return result

# def main():
#     # 웹캠 초기화
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 얼굴 감정 탐지
#         result = detect_emotion(frame)
        
#         # 감정 결과를 프레임에 표시
#         y_offset = 30
#         for res in result:
#             label = res['label']
#             score = res['score']
#             text = f'{label}: {score:.2f}'
#             cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             y_offset += 30

#         # 프레임을 화면에 표시
#         cv2.imshow("Webcam Emotion Detection", frame)

#         # 'q' 키를 누르면 루프 종료
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 자원 해제
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

# 감정 분류 파이프라인 설정
classifier = pipeline("image-classification", model="touchtech/fashion-images-gender-age-vit-large-patch16-224-in21k-v2")

# 이미지 크기 조정 함수
def resize_image(image_path, width):
    # 이미지 파일을 PIL 이미지로 로드
    pil_img = Image.open(image_path)

    # 이미지 크기 조정
    wpercent = (width / float(pil_img.size[0]))
    hsize = int((float(pil_img.size[1]) * float(wpercent)))
    resized_img = pil_img.resize((width, hsize), Image.LANCZOS)

    return resized_img

# 감정 탐지 함수
def detect_emotion(image_path):
    # 이미지 크기 조정
    resized_img = resize_image(image_path, 800)

    # 이미지를 분류
    result = classifier(resized_img)

    return result, resized_img

def main():
    # 이미지 파일 경로
    image_path = r"C:\Users\user\dev\AI\gender\k1.jpg"  # 여기에 이미지 파일 경로를 입력하세요

    # 감정 탐지
    result, resized_img = detect_emotion(image_path)

    # PIL 이미지를 OpenCV 형식으로 변환
    frame = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
    
    # 감정 결과를 프레임에 표시
    y_offset = 30
    for res in result:
        label = res['label']
        score = res['score']
        text = f'{label}: {score:.2f}'
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 30

    # 프레임을 화면에 표시
    cv2.imshow("Image Emotion Detection", frame)

    # 'q' 키를 누르면 루프 종료
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
