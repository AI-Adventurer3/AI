# 웹캠 표정 실시간 인식
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

# 감정 분류 파이프라인 설정
classifier = pipeline("image-classification", model="trpakov/vit-face-expression")

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
