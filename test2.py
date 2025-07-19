import os
import librosa
import numpy as np
import soundfile as sf
import json
import yt_dlp

# Step 1. 유튜브에서 오디오 다운로드
def download_audio(youtube_url, output_path='original.wav'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'original.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    os.rename("original.wav", output_path)
    print(f"[다운로드 완료] 오디오 저장: {output_path}")

# Step 2. 음정(Pitch) & 박자(Onset) 분석
def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    # 음정 추출
    print(f"[분석 중] 음정 추출")
    f0, voiced_flag, _ = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )

    # 박자 추출
    print(f"[분석 중] 박자 추출")
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()  # <=== 리스트로 변환

    # 템포 추정
    print(f"[분석 중] 템포 추정")
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.array(tempo))  # 명시적 float 처리

    print(f"[분석 완료] 평균 템포: {tempo:.2f} BPM")

    return {
        "sample_rate": int(sr),
        "pitch_hz": [float(p) if p is not None and not np.isnan(p) else None for p in f0],  # NaN 제거 + float 변환
        "onset_times": onset_times,
        "tempo_bpm": tempo
    }

# Step 3. 결과 저장 (기준 JSON)
def save_as_json(data, filename='song_profile.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[저장 완료] 기준 데이터 → {filename}")

# 🎬 실행 예시
if __name__ == '__main__':
    # url = input("분석할 유튜브 URL을 입력하세요: ").strip()
    # download_audio(url)
    data = analyze_audio('audio.wav')
    save_as_json(data)
