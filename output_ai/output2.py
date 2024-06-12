from transformers import pipeline
import openai

# OpenAI API 키 설정
openai.api_key = ""

# 모델 초기화
obj_detector = pipeline("object-detection", model="nickmuchi/yolos-base-finetuned-masks")
face_classifier = pipeline("image-classification", model="AliGhiasvand86/gisha_coverd_uncoverd_face")
face_expression = pipeline("image-classification", model="trpakov/vit-face-expression")
gender_detector = pipeline("image-classification", model="dima806/man_woman_face_image_detection")
image_captioning = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", framework="pt")

# 이미지 파일 경로 설정
# image_path = "C:/Users/user/project/aiadventurer/AI/AI/output_ai/image/a.jpg"
image_path="k.jpg"

# 모델 실행
obj_detection_result = obj_detector(image_path)
face_result = face_classifier(image_path)
expression_result = face_expression(image_path)
gender_result = gender_detector(image_path)
caption_result = image_captioning(image_path)

# 얼굴 가림 여부 확인
face_status = "uncovered" if face_result[0]['score'] > 0.6 else "covered"

# 높은 신뢰도의 표정 및 성별 레이블 추출
expression_labels = [result['label'] for result in expression_result if result['score'] > 0.7]
gender_labels = [result['label'] for result in gender_result if result['score'] > 0.7]

# 간단한 요약 생성
summary = (
    f"{face_status} face "
    f"{gender_labels} "
    f"with {expression_labels} face "
    f"{caption_result[0]['generated_text']}"
)

# OpenAI GPT 모델을 사용하여 요약 문장 자연스럽게 만들기기기
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Make it short and sound natural and change it to Korean without [] {summary}"}
    ],
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

# 요약된 문장 출력
final_summary = response['choices'][0]['message']['content'].strip()
print("얼굴 인식 결과:", face_status)
print("표정 인식 결과:", expression_labels)
print("성별 감지 결과:", gender_labels)
print("이미지 캡셔닝 결과:", caption_result)
print("요약 문장:", summary)
print("자연스러운 요약 문장:", final_summary)
