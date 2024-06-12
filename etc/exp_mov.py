import cv2
from transformers import pipeline
from PIL import Image
import os

# 세 개의 모델 설정
emotion_classifier = pipeline("image-classification", model="trpakov/vit-face-expression")
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", framework="pt")
gender_classifier = pipeline("image-classification", model="dima806/man_woman_face_image_detection")

def detect_emotion(frame):
    # 프레임을 PIL 이미지로 변환
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # 감정을 분류
    result = emotion_classifier(pil_img)
    return result

def generate_caption(frame):
    # 프레임을 PIL 이미지로 변환
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # 이미지를 캡션 생성
    result = captioner(pil_img)
    return result

def detect_gender(frame):
    # 프레임을 PIL 이미지로 변환
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # 성별을 분류
    result = gender_classifier(pil_img)
    return result

def main():
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 감정 탐지, 이미지 캡션 생성, 성별 구분
        emotion_result = detect_emotion(frame)
        caption_result = generate_caption(frame)
        gender_result = detect_gender(frame)

        # 감정 결과를 프레임에 표시
        y_offset = 30
        for res in emotion_result:
            label = res['label']
            score = res['score']
            text = f'Emotion - {label}: {score:.2f}'
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30
            
            # 감정이 sad, angry, fear, disgust 중 하나일 경우 화면 캡처
            if label in ['sad', 'angry', 'fear', 'disgust'] and score >= 0.6 :
                cv2.imwrite('captured_frame.jpg', frame)
                capture_path = os.path.abspath('captured_frame.jpg')
                
                # 성별, 감정, 행동을 한 문장으로 출력
                gender_label = gender_result[0]['label']
                caption_text = caption_result[0]['generated_text']
                print(f" {gender_label}이,  {label}표정으로,  {caption_text}을 했습니다")

        # 이미지 캡션 결과를 프레임에 표시
        if caption_result:
            caption = caption_result[0]['generated_text']
            cv2.putText(frame, f'Caption: {caption}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            y_offset += 30

        # 성별 결과를 프레임에 표시
        for res in gender_result:
            label = res['label']
            score = res['score']
            text = f'Gender - {label}: {score:.2f}'
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            y_offset += 30

        # 프레임을 화면에 표시
        cv2.imshow("Webcam Emotion, Captioning, and Gender Detection", frame)

        # 'q' 키를 누르면 루프 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
      
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

