
from flask import Flask, request, send_file, jsonify, send_from_directory
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
import shutil
import tempfile
import psola
import soundfile as sf
from flask_cors import CORS, cross_origin
import math
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from googleapiclient.discovery import build
import whisper
import subprocess
import sys
import zipfile
import json

load_dotenv()
api_key = os.getenv("API_KEY")

app=Flask(__name__)
CORS(app, resources={r"/api/*": {"origins" : "*"}})
app.debug=True
gpuserver="http://172.20.12.17:80"
# def set_oom_score_adj(value: int):
#     """
#     현재 프로세스의 oom_score_adj 값을 변경합니다.
#     value: -1000(절대 종료 금지) ~ +1000(우선 킬) 범위
#     """
#     path = "/proc/self/oom_score_adj"
#     try:
#         with open(path, "w") as f:
#             f.write(str(value))
#         print(f"OOM_SCORE_ADJ 설정 완료: {value}")
#     except Exception as e:
#         print(f"OOM_SCORE_ADJ 설정 실패: {e}")

# set_oom_score_adj(-1000)
def get_pitch_feedback(user_feature, singer_feature):
    # Flask 서버 주소와 포트(자기 자신, 혹은 네트워크 주소)
    # 로컬 서버 구동 중이면 이렇게:
    api_url = "http://172.20.12.17:80/llm_pitch_feedback"
    # 만약 Docker나 외부 서버면 실제 주소로 교체

    payload = {
        "user_feature": user_feature,
        "singer_feature": singer_feature
    }

    try:
        resp = requests.post(api_url, json=payload)
        resp.raise_for_status()
        result = resp.json()
        return result.get("feedback")
    except Exception as e:
        # 오류시 None이나 대신 에러 메시지 반환
        return f"Error: {e}"
    
def send_to_separation_api(input_path, save_dir):
    """
    input_path: 분리할 원본 오디오 파일
    save_dir: zip의 결과를 풀 디렉토리(outputs/<uuid>/)
    """
    url = gpuserver+'/separate'
    
    # 1. POST로 분리 요청 (파일 업로드)
    with open(input_path, 'rb') as f:
        response = requests.post(url, files={'audio': (os.path.basename(input_path), f)})
    
    if response.status_code != 200 or not response.content:
        raise Exception('분리 API 호출 실패 or 결과 없음')
    
    # 2. zip 임시 저장
    zip_tmp_path = os.path.join(save_dir, 'separated.zip')
    with open(zip_tmp_path, 'wb') as f:
        f.write(response.content)
    
    # 3. 압축 해제
    with zipfile.ZipFile(zip_tmp_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)

    os.remove(zip_tmp_path)  # zip파일 삭제

    # 4. 경로 return (vocals.wav, accompaniment.wav 등)
    vocals_path = os.path.join(save_dir, 'vocals.wav')
    accomp_path = os.path.join(save_dir, 'accompaniment.wav')
    return vocals_path, accomp_path

def analyze_audio_via_gpu_api(file_path, gpu_api_url=gpuserver+"/analyze"):
    import requests
    with open(file_path, 'rb') as f:
        res = requests.post(gpu_api_url, files={'audio': (os.path.basename(file_path), f)})
    if res.status_code == 200:
        return res.json()
    else:
        raise Exception(f'GPU analyze API 오류: {res.status_code}, {res.text}')
    


#유튜브 영상 다운(지정 폴더 안에 mp3 형식으로로)
def download_audio_with_ytdlp(youtube_url, filename, folder):
    ydl_opts = {
        'format': 'bestaudio/best',  # 최고 품질의 오디오 형식 지정
        'extractaudio': True,  # 오디오만 추출
        'audioformat': 'mp3',  # 오디오 형식을 wav로 지정
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



NOTE_NAMES = ["도", "도#", "레", "레#", "미", "파", "파#", "솔", "솔#", "라", "라#", "시"]

def hz_to_note_name(hz):
    if hz is None or hz <= 0:
        return None
    midi_num = round(69 + 12 * math.log2(hz / 440))
    octave = (midi_num // 12) - 1
    name = NOTE_NAMES[midi_num % 12]
    return f"{name}{octave}"

@app.route("/")
def index():
    return "hello"


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
@cross_origin(origin='http://localhost:3000')
def accompaniment_only():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']

    # 1. uuid 생성
    musicid = str(uuid.uuid4())

    # 2. 파일명을 uuid.mp3로 uploads에 저장
    upload_dir = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_dir, exist_ok=True)
    input_path = os.path.join(upload_dir, f"{musicid}.mp3")
    file.save(input_path)

    # 3. outputs/uuid 폴더 생성 후 분리
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], musicid)
    os.makedirs(output_dir, exist_ok=True)
    vocals_path, accomp_path = send_to_separation_api(input_path, output_dir)


    accomp_mp3 = accomp_path.replace('.wav', '.mp3')
    vocals_mp3 = vocals_path.replace('.wav', '.mp3')

    if not os.path.exists(accomp_mp3) and os.path.exists(accomp_path):
        AudioSegment.from_wav(accomp_path).export(accomp_mp3, format="mp3")
    if not os.path.exists(vocals_mp3) and os.path.exists(vocals_path):
        AudioSegment.from_wav(vocals_path).export(vocals_mp3, format="mp3")

    try:
        if os.path.exists(accomp_path): os.remove(accomp_path)
        if os.path.exists(vocals_path): os.remove(vocals_path)
        # if os.path.exists(accomp_mp3):
        #     os.remove(accomp_mp3)
        # if os.path.exists(vocals_mp3):
        #     os.remove(vocals_mp3)
        # os.remove(audio_dir)
    except Exception as e:
        print("삭제 실패:", e)

    # DB 처리 등 필요시 추가
    data = analyze_audio_via_gpu_api(
        vocals_mp3,
        gpu_api_url=gpuserver+"/analyze"  # 실제 GPU 서버 주소로 바꿔주세요
    )
    conn = get_conn()
    cur = conn.cursor()
    print(data)
    cur.execute(
        "INSERT INTO music_meta (musicid, pitch_vector, onset_times, lyrics) VALUES (%s, %s, %s, %s) RETURNING *",
        (musicid, data['pitch_hz'], data['onset_times'], json.dumps(data['lyrics']))
    )
    cur.execute("INSERT INTO musics (musicid, accompaniment_path) VALUES (%s, %s) RETURNING *",
        (musicid, accomp_mp3))
    
    #cur.execute()
    conn.commit()
    cur.close()
    conn.close()

    
        
    # uuid 반환
    return jsonify({'uuid': musicid}), 200

#post 유튜브 링크 form으로 받음, uuid로 음악 id 정하고 uuid로 반환, musics db에 곡정보없이 들어감, music_meta에 분석결과도 들어감감
@app.route('/accompaniment_with_ytlink', methods=['POST'])
def accompaniment_ytlink():

    youtube_url = request.form['youtube_url']

    filename=str(uuid.uuid4())
    file_name = download_audio_with_ytdlp(youtube_url, filename, "uploads/")
    input_path = file_name

    # 분리
    output_dir = os.path.join("outputs", filename)
    os.makedirs(output_dir, exist_ok=True)
    vocals_path, accomp_path = send_to_separation_api(input_path, output_dir)

    accomp_mp3 = accomp_path.replace('.wav', '.mp3')
    vocals_mp3 = vocals_path.replace('.wav', '.mp3')
    if not os.path.exists(accomp_mp3) and os.path.exists(accomp_path):
        AudioSegment.from_wav(accomp_path).export(accomp_mp3, format="mp3")
    if not os.path.exists(vocals_mp3) and os.path.exists(vocals_path):
        AudioSegment.from_wav(vocals_path).export(vocals_mp3, format="mp3")

    data = analyze_audio_via_gpu_api(
        vocals_mp3,
        gpu_api_url = gpuserver+"/analyze"  # 실제 GPU 서버 주소로 바꿔주세요
    )
    conn = get_conn()
    cur = conn.cursor()
    print(data)
    cur.execute(
        "INSERT INTO music_meta (musicid, pitch_vector, onset_times, lyrics) VALUES (%s, %s, %s, %s) RETURNING *",
        (filename, data['pitch_hz'], data['onset_times'], json.dumps(data['lyrics']))
    )
    cur.execute("INSERT INTO musics (musicid, accompaniment_path) VALUES (%s, %s) RETURNING *",
        (filename, accomp_mp3))
    
    #cur.execute()
    conn.commit()
    cur.close()
    conn.close()

    try:
        if os.path.exists(accomp_path):
            os.remove(accomp_path)
        if os.path.exists(vocals_path):
            os.remove(vocals_path)
        # if os.path.exists(accomp_mp3):
        #     os.remove(accomp_mp3)
        # if os.path.exists(vocals_mp3):
        #     os.remove(vocals_mp3)
        # os.remove(audio_dir)
    except Exception as e:
        print("삭제 실패:", e)

    return jsonify({'uuid': filename}), 200


#GET으로 uuid 받아서 반주 mp3 줌줌
@app.route('/get_accompaniment', methods=['GET'])
def get_accomp():
    musicid = request.args.get('musicid')  # 예: /get_accompaniment?musicid=xxxx-uuid
    if not musicid:
        return "musicid query param is required", 400

    # 경로 구성 (예시: 'outputs/xxxx-uuid/accompaniment.mp3')
    accomp_mp3 = os.path.join("outputs", musicid, "accompaniment.mp3")
    if not os.path.exists(accomp_mp3):
        return "mp3 not found", 404

    return send_file(
        accomp_mp3,
        as_attachment=True,
        download_name=f"{musicid}.mp3",
        mimetype='audio/mp3'
    )


#patch 요청, musicid 받아서 곡정보 수정(제목, 아티스트트)
@app.route('/musics/<musicid>', methods=['PATCH'])
def patch_music(musicid):
    req_data = request.get_json()
    title = req_data.get('title')
    artist = req_data.get('artist')
    genre = req_data.get('genre')

    if title is None and artist is None:
        return jsonify({'error': '수정할 항목이 없습니다.'}), 400

    # UPDATE 쿼리 만들기 (동적으로)
    fields = []
    values = []

    if title is not None:
        fields.append("title = %s")
        values.append(title)
    if artist is not None:
        fields.append("artist = %s")
        values.append(artist)
    if artist is not None:
        fields.append("genre = %s")
        values.append(genre)

    values.append(musicid)  # WHERE 조건

    set_clause = ', '.join(fields)
    query = f"UPDATE musics SET {set_clause} WHERE musicid = %s RETURNING *"

    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(query, values)
        updated = cur.fetchone()
        conn.commit()
        if not updated:
            return jsonify({'error': 'Not found'}), 404
        return jsonify({'musicid': updated[0], 'title': updated[1], 'artist': updated[2], 'genre': updated[3]}), 200
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()

#patch요청, userid 받아서 유저 정보 수정정
@app.route('/users/<userid>', methods=['PATCH'])
def patch_user(userid):
    req = request.get_json()
    nickname = req.get('nickname')
    profile_url = req.get('profile_url')
    is_online = req.get('is_online')

    if nickname is None and profile_url is None and is_online is None:
        return jsonify({'error': '수정할 항목이 필요합니다.'}), 400

    fields = []
    values = []

    if nickname is not None:
        fields.append('nickname = %s')
        values.append(nickname)
    if profile_url is not None:
        fields.append('profile_url = %s')
        values.append(profile_url)
    if is_online is not None:
        fields.append('is_online = %s')
        values.append(is_online)

    values.append(userid)

    set_clause = ', '.join(fields)
    query = f"UPDATE users SET {set_clause} WHERE userid = %s RETURNING *"

    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(query, values)
        updated = cur.fetchone()
        conn.commit()
        if not updated:
            return jsonify({'error': 'Not found'}), 404
        return jsonify({
            'userid': updated[0], 
            'nickname': updated[2],
            'profile_url': updated[3],
            'is_online': updated[4]
        }), 200
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()

#GET 요청, userid 받아서 유저 모든 정보 반환환
@app.route('/users/<userid>', methods=['GET'])
def get_user(userid):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute('SELECT userid, passwd, nickname, profile_url, is_online, created_at FROM users WHERE userid = %s', (userid,))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "User not found"}), 404
        # 필드명 순서에 맞춰 딕셔너리로 변환
        user = {
            "userid": row[0],
            "passwd": row[1],
            "nickname": row[2],
            "profile_url": row[3],
            "is_online": row[4],
            "created_at": row[5].isoformat() if row[5] else None  # datetime → str 변환
        }
        return jsonify(user)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


#GET 요청, musicid 받아서 음악 분석한 결과 반환 (onset, pitch)
@app.route('/music_meta/<musicid>', methods=['GET'])
def get_music_meta(musicid):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            'SELECT * FROM music_meta WHERE musicid = %s',
            (musicid,)
        )
        row = cur.fetchone()
        if not row:
            return jsonify({'error': 'Not found'}), 404
        music_meta = {
            'musicid': row[0],
            'pitch_vector': row[1],   # FLOAT8[] 타입은 psycopg2가 바로 파이썬 리스트로 반환
            'onset_times': row[2],
            'lyrics': row[3]
        }
        return jsonify(music_meta), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()
        

@app.route('/music_meta_note/<musicid>', methods=['GET'])
def get_music_meta_note(musicid):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            'SELECT pitch_vector FROM music_meta WHERE musicid = %s',
            (musicid,)
        )
        row = cur.fetchone()
        if not row:
            return jsonify({'error': 'Not found'}), 404
        pitch=row[0]

        note_arr = [hz_to_note_name(hz) for hz in pitch]

        return jsonify({'notes': note_arr}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


#GET 요청, 파라미터 없음. 현재기준 지니 TOP 50 JSON 반환환
@app.route('/genie-chart', methods=['GET'])
def genie_chart():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'
    }
    url = 'https://www.genie.co.kr/chart/top200?ditc=D&hh=23&rtm=N&pg=1'
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    

    music_list = soup.select('#body-content > div.newest-list > div > table > tbody > tr')
    print(music_list)
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
    
    
#PATCH 요청, title, artist, musicid 받아서 musics 업데이트트
@app.route('/edit-music', methods=['PATCH'])
def add_music():
    data = request.json
    title = data.get('title', 'untitled')
    artist = data.get('artist', 'unknown')
    genre = data.get('genre', 'unknown')
    music_id=data['musicid']

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE musics SET title=%s, artist=%s, genre=%s WHERE musicid=%s RETURNING musicid",
        (title, artist, genre, music_id)
    )
    music_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'id': music_id, 'message': 'Music added!'})

#POST 요청, userid, passwd, nickname 받아서 users에 유저 추가 
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

@app.route('/add_echo', methods=['POST'])
def add_echo():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']

    # 숫자 파라미터 가져오기 (없으면 기본값)
    try:
        delay_ms = int(request.form.get("delay", 250))      # ms
        repeat = int(request.form.get("repeat", 3))         # 횟수
        decay = float(request.form.get("decay", 0.5))       # 0~1
    except Exception as e:
        return jsonify({'error': 'Invalid parameter type'}), 400

    # 임시폴더에 파일 저장
    tmpdir = tempfile.mkdtemp()
    input_path = os.path.join(tmpdir, "input.mp3")
    file.save(input_path)

    # 오디오 로드
    audio = AudioSegment.from_file(input_path)

    # 에코 믹스
    output = audio
    for i in range(1, repeat+1):
        # 감쇠(dB) 계산: decay=0.5-> -6dB, decay=0.8-> -1.9dB
        attenuation = 20 * (i) * (0 if decay == 1 else np.log10(decay))
        echo = audio.apply_gain(attenuation)
        # 지연시킨 후 overlay
        echo = AudioSegment.silent(duration=delay_ms * i) + echo
        output = output.overlay(echo)

    # mp3 저장 및 전송
    result_path = os.path.join(tmpdir, "output.mp3")
    output.export(result_path, format="mp3")
    return send_file(
        result_path,
        as_attachment=True,
        download_name="echoed.mp3",
        mimetype="audio/mp3"
    )

@app.route('/autotune_vocal', methods=['POST'])
def autotune_vocal():
    musicid = request.form.get('musicid')
    if not musicid:
        return jsonify({'error': 'musicid required'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    # 1. DB에서 pitch_hz 가져오기
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT pitch_vector FROM music_meta WHERE musicid = %s", (musicid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return jsonify({'error': 'musicid not found in music_meta'}), 404

    pitch_arr_raw = row[0]   # 보통 psycopg2가 float8[]를 list로 변환
    # None→np.nan
    pitch_arr = np.array([x if x is not None else np.nan for x in pitch_arr_raw], dtype=float)

    # 2. 임시 저장 및 오디오 로드
    tmpdir = tempfile.mkdtemp()
    audio_path = os.path.join(tmpdir, 'input.mp3')
    audio_file.save(audio_path)
    y, sr = librosa.load(audio_path, sr=22050)

    # 3. 길이 맞춰 자르기
    frame_length = 2048
    hop_length = 256
    n_frames = int((len(y) - frame_length) // hop_length) + 1
    target_pitch = pitch_arr[:n_frames]
    y = y[:(n_frames * hop_length + frame_length)]

    # 4. 오토튠
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')
    y_tuned = psola.vocode(y, sample_rate=sr, target_pitch=target_pitch, fmin=fmin, fmax=fmax)

    # 5. 결과 저장/반환
    tuned_wav_path = os.path.join(tmpdir, "tuned.wav")
    tuned_mp3_path = os.path.join(tmpdir, "tuned.mp3")
    sf.write(tuned_wav_path, y_tuned, sr)
    AudioSegment.from_wav(tuned_wav_path).export(tuned_mp3_path, format="mp3")

    return send_file(
        tuned_mp3_path,
        as_attachment=True,
        download_name="tuned_vocal.mp3",
        mimetype="audio/mp3"
    )

@app.route('/ytlink/<query>', methods=['GET'])
def get_ytlink(query):
    youtube = build('youtube', 'v3', developerKey=api_key)

    request = youtube.search().list(
        q=query, part="snippet", type="video", maxResults=1
    )
    response = request.execute()
    video_id = response["items"][0]["id"]["videoId"]
    url = f"https://www.youtube.com/watch?v={video_id}"
    return jsonify({'link': url, 'message': 'link is in link!'})

@app.route('/user_record', methods=['POST'])
def user_record():
    # 1. 입력값 파싱
    userid = request.form.get('userid')
    musicid = request.form.get('musicid')
    score = request.form.get('score')
    if not (userid and musicid and score):
        return jsonify({'error': 'userid, musicid, score required'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file required'}), 400

    audio_file = request.files['audio']
    try:
        os.makedirs(os.path.join('records', userid), exist_ok=True)
        audio_path = os.path.join('records', userid, f"{musicid}.mp3")
        audio_file.save(audio_path)
    except Exception as e:
        return jsonify({'error': f'File save failed: {e}'}), 500

    # 2. pitch_vector, onset_times 분석
    #data = analyze_audio(audio_path)
    data = analyze_audio_via_gpu_api(
        audio_path,
        gpu_api_url = gpuserver+"/analyze"  # 실제 GPU 서버 주소로 바꿔주세요
    )
    # 3. DB 저장
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_records (userid, musicid, score, audio_url, pitch_vector, onset_times)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING recordid;
            """,
            (
                userid,
                musicid,
                float(score),
                audio_path,
                data['pitch_hz'],
                data['onset_times']
            )
        )
        recordid = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        return jsonify({'error': f'DB insert failed: {e}'}), 500

    return jsonify({
        'recordid': recordid,
        'audio_url': audio_path,
    }), 200

@app.route('/challenge', methods=['POST'])
def add_challenge():
    title = request.form.get('title')
    descript = request.form.get('descript')
    if not title:
        return jsonify({'error': 'title required'}), 400

    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO challenges (title, descript)
            VALUES (%s, %s)
            RETURNING challengeid;
        """, (title, descript))
        challengeid = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'challengeid': challengeid, 'title': title, 'descript': descript}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/user_challenge', methods=['POST'])
def add_user_challenge():
    userid = request.form.get('userid')
    challengeid = request.form.get('challengeid')
    if not userid or not challengeid:
        return jsonify({'error': 'userid and challengeid required'}), 400

    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_challenges (userid, challengeid)
            VALUES (%s, %s)
            RETURNING usrchalid;
        """, (userid, challengeid))
        usrchalid = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'usrchalid': usrchalid, 'userid': userid, 'challengeid': challengeid}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/lyrics/<musicid>', methods=['GET'])
def get_lyrics(musicid):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT lyrics FROM music_meta WHERE musicid = %s", (musicid,))
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    if not row or not row[0]:
        return jsonify({'error': f'lyrics for musicid {musicid} not found'}), 404

    return jsonify({'musicid': musicid, 'lyrics': row[0]})



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
@cross_origin(origin='http://localhost:3000')
def upload_file():
    if 'file' not in request.files or 'userid' not in request.form:
        return jsonify({'error': 'No file or userid'}), 400

    file = request.files['file']
    userid = request.form['userid']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    # 확장자 추출 후 <userid>.<확장자>로 저장
    ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
    filename = f'{secure_filename(userid)}.{ext}'
    user_dir = 'users'
    os.makedirs(user_dir, exist_ok=True)
    filepath = os.path.join(user_dir, filename)

    # 같은 이름의 파일이 있으면 덮어쓰기
    file.save(filepath)
    profile_url = f'{user_dir}/{filename}'

    return jsonify({'profile_url': profile_url})



@app.route('/profile/<userid>', methods=['GET'])
def get_profile(userid):
    user_dir = 'users'
    # 지원 확장자 순회하며 파일 탐색
    for ext in ALLOWED_EXTENSIONS:
        filename = f"{secure_filename(userid)}.{ext}"
        filepath = os.path.join(user_dir, filename)
        if os.path.exists(filepath):
            # send_from_directory로 이미지 반환
            return send_from_directory(user_dir, filename)
    # 파일이 없으면 404
    return jsonify({'error': 'No profile image found for this user'}), 404

@app.route('/check_userid', methods=['POST'])
def check_userid():
    req_data = request.get_json()
    userid = req_data.get('userid')
    if not userid:
        return jsonify({"error": "userid not provided"}), 400

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE userid = %s", (userid,))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()

    return jsonify({
        "userid": userid,
        "exists": exists  # True(이미 있음) / False(사용 가능)
    })

def process_voice_features(pitch_vector):
    # None/NaN이나 0이 아닌 값만 남김
    pitch_valid = np.array([f for f in pitch_vector if f is not None and not np.isnan(f) and f > 0])
    if len(pitch_valid) == 0:
        return None

    # 기본 통계
    pitch_mean = float(np.mean(pitch_valid))
    pitch_std = float(np.std(pitch_valid))

    # jitter (평소대를 기준으로 프레임별 차이의 평균)
    if len(pitch_valid) > 1:
        diffs = np.abs(np.diff(pitch_valid))
        jitter_percent = float(np.mean(diffs / pitch_valid[:-1])) * 100
    else:
        jitter_percent = 0.0

    # voiced_ratio (음이 검출된 프레임 비율, 전체 벡터 중 not None/NaN/0)
    voiced_ratio = float(len(pitch_valid) / len(pitch_vector))

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "voiced_ratio": voiced_ratio,
        "jitter_percent": jitter_percent
    }

@app.route('/vocal_assessment', methods=['POST'])
def voice_features():
    data = request.get_json()
    userid = data.get('userid')
    musicid = data.get('musicid')
    if not userid or not musicid:
        return jsonify({'error': 'userid, musicid required'}), 400

    conn = get_conn()
    cur = conn.cursor()

    # 가장 최근 user pitch_vector SELECT
    cur.execute("""
        SELECT pitch_vector FROM user_records
        WHERE userid = %s AND musicid = %s
        ORDER BY created_at DESC LIMIT 1
    """, (userid, musicid))
    user_row = cur.fetchone()

    # 가수(원곡) pitch_vector SELECT
    cur.execute("""
        SELECT pitch_vector FROM music_meta WHERE musicid = %s
    """, (musicid,))
    singer_row = cur.fetchone()

    cur.close()
    conn.close()

    if not user_row or not singer_row:
        return jsonify({'error': 'user 또는 music 데이터 없음'}), 404

    user_pitch = user_row[0]
    singer_pitch = singer_row[0]

    n = min(len(user_pitch), len(singer_pitch))
    user_pitch=user_pitch[:n]
    singer_pitch=singer_pitch[:n]

    # voice feature 계산
    user_features = process_voice_features(user_pitch)
    singer_features = process_voice_features(singer_pitch)
    if user_features is None or singer_features is None:
        return jsonify({'error': '피치 데이터 부족'}), 400
    feedback=get_pitch_feedback(user_features,singer_features)

    return jsonify({
        "feedback": feedback
    })

@app.route('/search_music', methods=['POST'])
def search_music():
    data = request.get_json()
    title = data.get('title')
    artist = data.get('artist')
    if not title or not artist:
        return jsonify({'error': 'title, artist 필요'}), 400

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT musicid, title, artist, genre, accompaniment_path, created_at
        FROM musics
        WHERE title = %s AND artist = %s
        LIMIT 1
    """, (title, artist))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return jsonify({'result': None, 'message': 'No matching music found'}), 404

    keys = ['musicid', 'title', 'artist', 'genre', 'accompaniment_path', 'created_at']
    music_info = dict(zip(keys, row))
    return jsonify({'result': music_info})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)