# 1 모듈 가져오기
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np
import io

# 2 추론기 만듬 이미지 캡션 파이프라인 설정
classifier = pipeline("image-classification", model="AliGhiasvand86/gisha_coverd_uncoverd_face", framework="pt")

def generate_classification(frame):
    # 프레임을 PIL 이미지로 변환
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 이미지를 분류
    result = classifier(pil_img)
    
    return result

def main():
    # 3 데이터 가져오기
    # 웹캠에서 실시간으로 이미지를 가져옵니다..
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 4: 추론
    # 웹캠에서 가져온 이미지에 대해 분류 수행
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 분류 생성
        result = generate_classification(frame)
        
        # 분류 결과를 프레임에 표시
        if result and len(result) > 0:
            if result[0]['label'] == 'uncovered' and result[0]['score'] > result[1]['score']:
                text = "uncovered"
            else:
                text = "covered"
        else:
            text = "No action detected"

        # 프레임에 텍스트를 표시
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 프레임을 화면에 표시
        cv2.imshow("Webcam Image Classification", frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5: 후처리 및 출력
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()