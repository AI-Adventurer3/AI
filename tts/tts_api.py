from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from gtts import gTTS
import os

app = FastAPI()

@app.post("/convert")
def convert_text_to_speech(text: str = Form(...)):
    # 텍스트를 음성으로 변환
    tts = gTTS(text=text, lang='ko')
    
    # 변환된 음성을 파일로 저장
    file_path = "output.mp3"
    tts.save(file_path)
    
    return FileResponse(file_path, media_type='audio/mpeg', filename="output.mp3")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, Form
# from fastapi.responses import StreamingResponse
# from gtts import gTTS
# import io

# app = FastAPI()

# @app.post("/convert")
# def convert_text_to_speech(text: str = Form(...)):
#     # 텍스트를 음성으로 변환
#     tts = gTTS(text=text, lang='ko')
    
#     # 변환된 음성을 메모리 스트림에 저장
#     audio_stream = io.BytesIO()
#     tts.write_to_fp(audio_stream)
#     audio_stream.seek(0)  # 스트림의 시작점으로 이동
    
#     headers = {
#         "Content-Disposition": "inline; filename=output.mp3"
#     }
    
#     return StreamingResponse(audio_stream, media_type="audio/mpeg", headers=headers)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

