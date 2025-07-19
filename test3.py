import matplotlib.pyplot as plt
import librosa
import os
import librosa
import numpy as np
import soundfile as sf
import json
import yt_dlp
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

def plot_pitch_onset(audio_path, result):
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa.display

    y, sr = librosa.load(audio_path, sr=22050)
    plt.figure(figsize=(14, 6))
    librosa.display.waveshow(y, sr=sr, color='gray', alpha=0.5)
    
    # numpy 배열로 바꿔서 times_like에 전달
    pitch_arr = np.array(result['pitch_hz'])
    times = librosa.times_like(pitch_arr, sr=sr)
    plt.plot(times, pitch_arr, label='Pitch (Hz)', color='b')

    # onset 지점 표시
    #for t in result['onset_times']:
        #plt.axvline(x=t, color='r', linestyle='--', alpha=0.7, label='Onset' if t == result['onset_times'][0] else "")
    
    plt.title(f"Pitch & Onset Graph\nTempo: {result['tempo_bpm']:.1f} BPM")
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (Hz)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("pitch_onset2.png")  # 외부 VM에서는 show() 대신 저장
    print("플롯이 pitch_onset2.png로 저장되었습니다.")


# 사용법 예시
if __name__ == '__main__':
    # analysis_data = analyze_audio('audio.wav')
    # save_as_json(analysis_data)
    analysis_data = analyze_audio('audio/vocals.wav')
    plot_pitch_onset('audio/vocals.wav', analysis_data)
