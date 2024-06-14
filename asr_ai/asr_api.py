import os
import wave
import json
import pyaudio
import numpy as np
import torch
import io
import soundfile as sf
from vosk import Model, KaldiRecognizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 초기화
app = FastAPI()

# Vosk 모델 로드
vosk_model_path = os.getenv("VOSK_MODEL_PATH", "C:\\Users\\user\\dev\\AI\\asr\\model")
vosk_model = Model(vosk_model_path)

# ASR 모델 로드
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# NLP 모델 로드
nlp_model = pipeline("sentiment-analysis")

def call_police():
    logger.info("Calling police...")
    # 실제 시스템 호출을 피하기 위해 주석 처리
    # os.system("echo 'Calling 911...'")

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    # 업로드된 오디오 파일 읽기
    contents = await file.read()

    # Vosk를 사용한 실시간 키워드 탐지
    recognizer = KaldiRecognizer(vosk_model, 16000)
    if recognizer.AcceptWaveform(contents):
        result = recognizer.Result()
        text = json.loads(result).get("text", "")
        logger.info(f"Detected speech: {text}")
        
        # 키워드 탐지
        if "help me" in text or "stop" in text or "hey" in text:
            logger.info("Keyword detected!")

            # 메모리에서 직접 처리
            audio_input, _ = sf.read(io.BytesIO(contents))
            input_values = asr_processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values
            with torch.no_grad():
                logits = asr_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = asr_processor.decode(predicted_ids[0])

            logger.info(f"Transcription: {transcription}")

            # NLP를 이용해 의도 파악
            intent = nlp_model(transcription)
            logger.info(f"Intent: {intent}")

            # 특정 명령어에 따라 행동 수행
            if "help me" in transcription:
                call_police()
                return JSONResponse(content={"message": "Police called based on transcription."})
    
    return JSONResponse(content={"message": "No keyword detected or no action required."})

# Uvicorn으로 앱 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
