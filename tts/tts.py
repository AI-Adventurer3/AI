# from transformers import pipeline

# pipe = pipeline("text-to-speech", model="YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")

# text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"

# example = dataset["test"][304]
# speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)

# forward_params = {"speaker_embeddings": speaker_embeddings}
# output = pipe(text, forward_params=forward_params)
# output


# from transformers import pipeline

# # text-to-speech 파이프라인 생성
# pipe = pipeline("text-to-speech", model="facebook/s2t-small-librispeech-asr")

# # 변환할 텍스트 지정
# text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"

# # 텍스트를 음성으로 변환
# output = pipe(text)

# # 변환된 음성 출력
# print(output)

from gtts import gTTS
import os

# 변환할 텍스트 지정
# text = "I'm learning coding in Gangnam. However, I'm actually a member of Marvel's Avengers. I'm now disguised to protect Seoul. And this is still unclear, but I think Loki is among the members who learn coding together."
text = "나는 강남에서 코딩을 배우고 있다. 하지만, 나는 사실 마블의 어벤져스 멤버이다. 나는 지금 서울을 보호하기 위해 위장을 하고 있다. 그리고 이것은 아직 확실하지 않은 정보이지만 같이 코딩 배우는 멤버들중에서 로키가 있는것 같다."


# 텍스트를 음성으로 변환
tts = gTTS(text=text, lang='ko')

# 변환된 음성을 파일로 저장
tts.save("output.mp3")

# 저장된 음성 파일 재생 (선택 사항)
os.system("start output.mp3")

print("텍스트가 음성으로 변환되었습니다. 'output.mp3' 파일을 확인하세요.")
