from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
import openai
import re
import io
from PIL import Image

# OpenAI API 키 설정
openai.api_key = ""

# 모델 초기화
obj_detector = pipeline("object-detection", model="nickmuchi/yolos-base-finetuned-masks")
face_classifier = pipeline("image-classification", model="AliGhiasvand86/gisha_coverd_uncoverd_face")
face_expression = pipeline("image-classification", model="trpakov/vit-face-expression")
image_captioning = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", framework="pt")
age_classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")

app = FastAPI()

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # 파일을 바이너리 데이터로 읽기
        byte_file = await file.read()

        # 바이너리 데이터를 PIL 이미지로 변환
        image_bin = io.BytesIO(byte_file)
        pil_img = Image.open(image_bin)

        # 이미지 분석 수행
        obj_detection_result = obj_detector(pil_img)
        face_result = face_classifier(pil_img)
        expression_result = face_expression(pil_img)
        caption_result = image_captioning(pil_img)
        age_result = age_classifier(pil_img)

        # 얼굴 가림 여부 확인
        face_status = "uncovered" if face_result[0]['score'] > 0.6 else "covered"

        # 표정 인식 결과에서 조건에 맞는 상위 두 개의 감정 레이블 추출
        expression_result_sorted = sorted(expression_result, key=lambda x: x['score'], reverse=True)
        top_two_expressions = []
        if expression_result_sorted[0]['score'] > 0.6:
            top_two_expressions.append(expression_result_sorted[0]['label'])
        if len(expression_result_sorted) > 1 and expression_result_sorted[1]['score'] > 0.4:
            top_two_expressions.append(expression_result_sorted[1]['label'])

        # 나이 예측 결과에서 두 번째로 높은 값 추출
        age_result_sorted = sorted(age_result, key=lambda x: x['score'], reverse=True)
        second_highest_age_label = age_result_sorted[1]['label']

        # 간단한 요약 생성
        summary = (
            f"얼굴을 {face_status}한 "
            f"이 {', '.join(top_two_expressions)} 표정으로 "
            f"나이는 {second_highest_age_label} "
            f"{caption_result[0]['generated_text']}"
        )

        # OpenAI GPT 모델을 사용하여 요약 문장을 자연스럽게 만들기
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

        # 폭력적이거나 위험한 문장인지 확인하는 함수
        def is_violent_or_dangerous(sentence):
            # 폭력적인 단어 리스트
            violent_keywords = ['주먹', '발길질', '때리다', '공격', '싸움', '폭력', '협박', '위협', '칼', '총', '무기', '당근']
            # 위험한 감정
            emotion_keywords = ['슬픈', '화난','두려운']
            # 위험한 상황 단어 리스트
            dangerous_keywords = ['불', '화재', '추락', '붕괴', '폭발', '전기', '피', '상처', '응급', '구조', '부상']

            # 정규 표현식을 사용하여 문장에서 폭력적이거나 위험한 단어가 있는지 확인
            for keyword in violent_keywords + emotion_keywords + dangerous_keywords:
                if re.search(keyword, sentence):
                    return True
            return False

        # 최종 요약 문장이 폭력적이거나 위험한지 확인
        is_dangerous = is_violent_or_dangerous(final_summary)

        return {
            "face_status": face_status,
            "expressions": top_two_expressions,
            "caption": caption_result[0]['generated_text'],
            "age": second_highest_age_label,
            "summary": final_summary,
            "is_dangerous": is_dangerous
        }
    except Exception as e:
        # 예외 발생 시 에러 로그 출력
        print(f"Error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
