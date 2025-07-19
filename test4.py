import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from spleeter.separator import Separator

# 1. Spleeter로 보컬 분리
def separate_vocals(input_audio, output_dir='outputs'):
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(input_audio, '.')
    base = os.path.splitext(os.path.basename(input_audio))[0]
    vocals_path = f"{output_dir}/{base}/audio.mp3"
    return vocals_path

# 2. 보컬 wav에서 pitch, onset 추출 및 그래프 생성
def plot_vocal_pitch_onset(vocal_wav, save_name="vocals_pitch_onset.png"):
    y, sr = librosa.load(vocal_wav, sr=22050)
    # 피치 분석
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    # 온셋(음 시작점) 탐지
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    # 그래프 그리기
    plt.figure(figsize=(14, 6))
    librosa.display.waveshow(y, sr=sr, color='gray', alpha=0.5)
    times = librosa.times_like(f0, sr=sr)
    plt.plot(times, f0, label='Pitch (Hz)', color='b')
    for idx, t in enumerate(onset_times):
        plt.axvline(x=t, color='r', linestyle='--', alpha=0.7, label='Onset' if idx == 0 else "")
    plt.title("보컬 피치 & 온셋 시각화")
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (Hz)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    print(f"그래프 이미지가 {save_name}로 저장되었습니다!")

# 3. 통합 실행 예시
if __name__ == '__main__':
    input_audio = 'audio.mp3'  # 분석할 원본 오디오 경로 (mp3/wav 등)
    vocals_wav = separate_vocals(input_audio)          # 보컬만 분리
    plot_vocal_pitch_onset(vocals_wav, "vocals_pitch_onset.png")   # 분석 및 그래프 저장
