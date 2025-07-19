
from flask import Flask, request, send_file, jsonify
import os
from spleeter.separator import Separator
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import requests
from bs4 import BeautifulSoup
from flask_sqlalchemy import SQLAlchemy
import yt_dlp
import uuid
import psycopg2
import librosa
import numpy as np

def download_audio_with_ytdlp(youtube_url, filename, folder):
    ydl_opts = {
        'format': 'bestaudio/best',  # 최고 품질의 오디오 형식 지정
        'extractaudio': True,  # 오디오만 추출
        'audioformat': 'wav',  # 오디오 형식을 wav로 지정
        'outtmpl': os.path.join(folder, f'{filename}.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',  # FFmpeg을 사용하여 오디오 추출
            'preferredcodec': 'mp3',  # 선호하는 코덱을 wav로 설정
            'preferredquality': '192',  # 오디오 품질을 192kbps로 설정
        }],
        'ffmpeg_location': '/usr/bin/ffmpeg',  # 필요한 경우 FFmpeg 경로 지정
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)  # YouTube 정보 추출 및 다운로드
        file_name = ydl.prepare_filename(info_dict).replace('.webm', '.mp3')  # 파일 이름을 wav로 변경
    return file_name  # 파일 이름 반환

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

    # # 템포 추정
    # print(f"[분석 중] 템포 추정")
    # tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # tempo = float(np.array(tempo))  # 명시적 float 처리

    # print(f"[분석 완료] 평균 템포: {tempo:.2f} BPM")

    return {
        "sample_rate": int(sr),
        "pitch_hz": [float(p) if p is not None and not np.isnan(p) else None for p in f0],  # NaN 제거 + float 변환
        "onset_times": onset_times,
        # "tempo_bpm": tempo
    }

app=Flask(__name__)
app.debug=True

@app.route("/")
def index():
    return "hello"

app = Flask(__name__)

def get_conn():
    return psycopg2.connect(
        dbname='musicdb',
        user='ksj',
        password='1234',
        host='localhost'
    )

UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

separator = Separator('spleeter:2stems')

@app.route('/accompaniment', methods=['POST'])
def accompaniment_only():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    file.save(input_path)
    # 분리
    separator.separate_to_file(input_path, app.config['OUTPUT_FOLDER'])
    base_filename = os.path.splitext(filename)[0]
    audio_dir = os.path.join(app.config['OUTPUT_FOLDER'], base_filename)
    accomp_wav = os.path.join(audio_dir, 'accompaniment.wav')
    accomp_mp3 = os.path.join(audio_dir, 'accompaniment.mp3')


    # wav → mp3 변환 (이미 있으면 변환 생략)
    if not os.path.exists(accomp_mp3):
        AudioSegment.from_wav(accomp_wav).export(accomp_mp3, format="mp3")
    return send_file(
        accomp_mp3,
        as_attachment=True,
        download_name=os.path.basename(accomp_mp3),
        mimetype='audio/mp3'
    )



@app.route('/accompaniment_with_ytlink', methods=['POST'])
def accompaniment_ytlink():

    youtube_url = request.form['youtube_url']


    filename=str(uuid.uuid4())
    file_name = download_audio_with_ytdlp(youtube_url, filename, "uploads/")
    input_path = file_name

    # 분리
    separator.separate_to_file(input_path, "outputs")

    base_filename = os.path.basename(file_name)
    audio_dir = os.path.splitext(os.path.join("outputs", base_filename))[0]
    accomp_wav = os.path.join(audio_dir, 'accompaniment.wav')
    accomp_mp3 = os.path.join(audio_dir, 'accompaniment.mp3')
    vocals_wav = os.path.join(audio_dir, 'vocals.wav')
    vocals_mp3 = os.path.join(audio_dir, 'vocals.mp3')

    #wav → mp3 변환 (이미 있으면 변환 생략)
    if not os.path.exists(accomp_mp3) and os.path.exists(accomp_wav):
        AudioSegment.from_wav(accomp_wav).export(accomp_mp3, format="mp3")
    if not os.path.exists(vocals_mp3) and os.path.exists(vocals_wav):
        AudioSegment.from_wav(vocals_wav).export(vocals_mp3, format="mp3")

    data=analyze_audio(accomp_mp3)
    conn = get_conn()
    cur = conn.cursor()
    print(data)
    cur.execute(
        "INSERT INTO music_meta (musicid, pitch_vector, onset_times) VALUES (%s, %s, %s) RETURNING *",
        (filename, data['pitch_hz'], data['onset_times'])
    )
    #result = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    try:
        if os.path.exists(accomp_wav):
            os.remove(accomp_wav)
        if os.path.exists(vocals_wav):
            os.remove(vocals_wav)
        # if os.path.exists(accomp_mp3):
        #     os.remove(accomp_mp3)
        # if os.path.exists(vocals_mp3):
        #     os.remove(vocals_mp3)
        # os.remove(audio_dir)
    except Exception as e:
        print("삭제 실패:", e)

    return send_file(
        accomp_mp3,
        as_attachment=True,
        download_name=f"{filename}.mp3",
        download_name=os.path.basename(accomp_mp3),
        mimetype='audio/mp3'
    )


@app.route('/genie-chart', methods=['GET'])
def genie_chart():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'
    }
    url = 'https://www.genie.co.kr/chart/top200?ditc=D&hh=23&rtm=N&pg=1'
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    music_list = soup.select('#body-content > div.newest-list > div > table > tbody > tr')

    result = []
    for music in music_list:
        rank = music.select_one('td.number').text[0:2].strip()
        title = music.select_one('td.info > a.title.ellipsis').text.strip()
        name = music.select_one('td.info > a.artist.ellipsis').text.strip()
        result.append({
            'rank': rank,
            'title': title,
            'artist': name
        })

    return jsonify(result)
    
    
@app.route('/add-music', methods=['POST'])
def add_music():
    data = request.json
    title = data['title']
    artist = data.get('artist', '')
    path = data.get('accompaniment_path', '')

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO musics (title, artist, accompaniment_path) VALUES (%s, %s, %s) RETURNING musicid",
        (title, artist, path)
    )
    music_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'id': music_id, 'message': 'Music added!'})

@app.route('/add-user', methods=['POST'])
def add_user():
    data = request.json
    userid = data['userid']
    passwd = data['passwd']
    nickname = data['nickname']
    profile_url = data.get('profile_url', '')
    is_online=True

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (userid, passwd, nickname, profile_url, is_online) VALUES (%s, %s, %s, %s, %s) RETURNING *",
        (userid, passwd, nickname, profile_url, is_online)
    )
    result = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return jsonify(result)


    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)