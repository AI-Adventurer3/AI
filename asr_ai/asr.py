import os
import wave
import json
import pyaudio
import numpy as np
import torch
import soundfile as sf
from vosk import Model, KaldiRecognizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline


# Vosk 모델 로드
vosk_model_path = "C:\\Users\\user\\dev\\AI\\asr\\model"
vosk_model = Model(vosk_model_path)

# ASR 모델 로드
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# NLP 모델 로드
nlp_model = pipeline("sentiment-analysis")

# 마이크에서 실시간 음성 데이터 읽기
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=16000,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=4000
)

def call_police():
    print("Calling police...")
    # 실제 시스템 호출을 피하기 위해 주석 처리
    # os.system("echo 'Calling 911...'")

# 실시간 키워드 탐지 및 행동 수행
recognizer = KaldiRecognizer(vosk_model, 16000)
try:
    while True:
        pcm = audio_stream.read(4000)
        if recognizer.AcceptWaveform(pcm):
            result = recognizer.Result()
            text = json.loads(result).get("text", "")
            print("Detected speech:", text)

            # 키워드 탐지
            if "help me" in text or "stop" in text or "hey" in text:
                print("Keyword detected!")

                # 음성 파일로 저장 후 재인식 (데모용, 실제로는 실시간 데이터를 바로 처리해야 함)
                with wave.open("temp_audio.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(16000)
                    wf.writeframes(pcm)

                # 저장된 음성 파일 로드
                audio_input, _ = sf.read("temp_audio.wav")
                input_values = asr_processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values
                with torch.no_grad():
                    logits = asr_model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = asr_processor.decode(predicted_ids[0])

                print("Transcription: ", transcription)

                # NLP를 이용해 의도 파악
                intent = nlp_model(transcription)
                print("Intent: ", intent)

                # 특정 명령어에 따라 행동 수행
                if "help me" in transcription:
                    call_police()
                break
finally:
    # 스트림 닫기
    audio_stream.stop_stream()
    audio_stream.close()
    pa.terminate()
