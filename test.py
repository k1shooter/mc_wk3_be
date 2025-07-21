import whisper
print("[분석 중] 가사 추출 (Whisper)")
model = whisper.load_model("base")   # "tiny", "base" 등 선택 가능
result = model.transcribe('outputs/8e57383a-43d9-474b-abe0-a70a0237d461/vocals.mp3', language='ko')
segments = result["segments"]
tt=""
for seg in segments:
    tt+=f"{seg['start']:.2f} ~ {seg['end']:.2f}:{seg['text']}/"

print(tt)
