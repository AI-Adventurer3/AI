# 1 모듈 가져오기
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

# 2 추론기 만듬 이미지 캡션 파이프라인 설정
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", framework="pt")

def generate_caption(frame):
    # 프레임을 PIL 이미지로 변환
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 이미지를 캡션 생성
    result = captioner(pil_img)
    
    return result

def main():
    # 3 데이터 가져오기
    # 웹캠에서 실시간으로 이미지를 가져옵니다.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 4: 추론
    # 웹캠에서 가져온 이미지에 대해 캡션 생성
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 캡션 생성
        result = generate_caption(frame)
        
        # 캡션 결과를 프레임에 표시
        y_offset = 30
        for res in result:
            caption = res['generated_text']
            cv2.putText(frame, caption, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30

        # 프레임을 화면에 표시
        cv2.imshow("Webcam Image Captioning", frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5: 후처리 및 출력
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
