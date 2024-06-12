from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import base64
from transformers import pipeline

app = FastAPI()

# 기존 모델들
obj_detector = pipeline("object-detection", model="nickmuchi/yolos-base-finetuned-masks")
face_classifier = pipeline("image-classification", model="AliGhiasvand86/gisha_coverd_uncoverd_face")

# 추가 모델들
face_expression = pipeline("image-classification", model="trpakov/vit-face-expression")
gender_detector = pipeline("image-classification", model="dima806/man_woman_face_image_detection")
image_captioning = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", framework="pt")

# HTML 템플릿
html_form = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>데이트 폭력</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f3f3f3;
            color: #333;
        }
        .container {
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }
        form {
            text-align: center;
        }
        input[type="file"] {
            display: none;
        }
        .custom-file-upload {
            display: inline-block;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 20px;
        }
        .result-container {
            margin-top: 30px;
        }
        .result {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        h2 {
            margin-top: 0;
            color: #444;
        }
        p {
            margin: 5px 0;
        }
    </style>
    <script>
        function previewImage(event) {
            var fileInput = event.target;
            var files = fileInput.files;
            if (files.length > 0) {
                var reader = new FileReader();
                reader.onload = function() {
                    var imgElement = document.getElementById('preview-image');
                    imgElement.src = reader.result;
                    document.getElementById('image-preview').style.display = 'block';
                }
                reader.readAsDataURL(files[0]);
            } else {
                document.getElementById('image-preview').style.display = 'none';
            }
        }
    </script>
</head>
<body>
    <h1>🚨데이트 폭력 감지!🚨</h1>
    <form action="/classify-image/" method="post" enctype="multipart/form-data">
        <label for="file-upload" class="custom-file-upload">파일 선택</label>
        <input id="file-upload" type="file" name="file" onchange="previewImage(event)">
        <input type="submit" value="분류하기">
    </form>
    <div class="result-container">
        %s
    </div>
    <div id="image-preview" style="display: none;">
        <h2>미리보기</h2>
        <img id="preview-image" src="#" alt="미리보기">
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_form % ""

@app.post("/classify-image/", response_class=HTMLResponse)
async def classify_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes))

    # 객체 감지 결과
    obj_results = obj_detector(image)
    # 얼굴 커버 분류 결과
    face_results = face_classifier(image)
    # 얼굴 표정 감지 결과
    face_exp_results = face_expression(image)
    # 성별 감지 결과
    gender_results = gender_detector(image)
    # 이미지 캡셔닝 결과
    caption_results = image_captioning(image)

    result_html = "<h2>업로드 이미지:</h2>"
    result_html += f'<img src="data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"/>'

    result_html += "<h2>마스크 상태:</h2>"
    for result in obj_results:
        score_percentage = f"{result['score'] * 100:.2f} % 확률로 씀"
        if result['label'] == 'with_mask':
            label = "마스크를 착용함"
        else:
            label = result['label']
        result_html += f"<p>{label}: {score_percentage}</p>"
    if not obj_results:
        result_html += "<p>마스크를 착용하지 않음</p>"

    result_html += "<h2>얼굴 상태:</h2>"
    for result in face_results:
        result_html += f"<p>{result['label']}: {result['score']:.2f}</p>"

    result_html += "<h2>얼굴 표정:</h2>"
    for result in face_exp_results:
        result_html += f"<p>{result['label']}: {result['score']:.2f}</p>"

    result_html += "<h2>성별 감지:</h2>"
    for result in gender_results:
        result_html += f"<p>{result['label']}: {result['score']:.2f}</p>"

    result_html += "<h2>이미지 설명:</h2>"
    for result in caption_results:
        result_html += f"<p>{result['generated_text']}</p>"

    return html_form % result_html

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
