import json
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import psola  # 본인 환경에 따라 import 수정

# (1) song_profile.json에서 pitch_hz 배열 불러오기
with open("hurtroadvocal.json", "r", encoding="utf-8") as f:
    sj = json.load(f)

pitch_arr = np.array([x if x is not None else np.nan for x in sj['pitch_hz']], dtype=float)

# (2) 오디오 로드
y, sr = librosa.load("hurtroadtest.mp3", sr=22050)

# (3) frame/hop 계산하고 pitch와 오디오 길이 맞추기
frame_length = 2048
hop_length = frame_length // 4
n_frames = int((len(y) - frame_length) // hop_length) + 1

target_pitch = pitch_arr[:n_frames]
y = y[:(n_frames * hop_length + frame_length)]

# (4) PSOLA로 튠
fmin = librosa.note_to_hz('C2')
fmax = librosa.note_to_hz('C7')
y_tuned = psola.vocode(y, sample_rate=sr, target_pitch=target_pitch, fmin=fmin, fmax=fmax)

# (5) 파일로 저장
sf.write("tunedhurtroad.wav", y_tuned, sr)
AudioSegment.from_wav("tunedhurtroad.wav").export("tunedhurtroad.mp3", format="mp3")

print("완료! tunedhurtroad.wav, tunedhurtroad.mp3 파일을 확인하세요.")
