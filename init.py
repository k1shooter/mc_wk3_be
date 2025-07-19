
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
import shutil
#유튜브 영상 다운(지정 폴더 안에 mp3 형식으로로)
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

#mp3 파일 분석, onset이랑 pitch
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
    separator.separate_to_file(input_path, output_dir)

    # 분리툴이 실제로 결과를 출력하는 위치: outputs/uuid/uuid 폴더  
    actual_dir = os.path.join(output_dir, musicid)   # outputs/uuid/uuid

    # 파일들을 상위로 이동
    accomp_wav_src = os.path.join(actual_dir, 'accompaniment.wav')
    vocals_wav_src = os.path.join(actual_dir, 'vocals.wav')
    accomp_wav_dst = os.path.join(output_dir, 'accompaniment.wav')
    vocals_wav_dst = os.path.join(output_dir, 'vocals.wav')

    if os.path.exists(accomp_wav_src):
        shutil.move(accomp_wav_src, accomp_wav_dst)
    if os.path.exists(vocals_wav_src):
        shutil.move(vocals_wav_src, vocals_wav_dst)
    # (필요시 기타 stem도 이동)

    # 중간 디렉토리 정리
    if os.path.exists(actual_dir):
        shutil.rmtree(actual_dir)

    # 이후부턴 항상 outputs/uuid/ 바로 밑에서 파일 처리 가능!
    accomp_wav= os.path.join(output_dir, 'accompaniment.wav')
    vocals_wav = os.path.join(output_dir, 'vocals.wav')
    accomp_mp3 = os.path.join(output_dir, 'accompaniment.mp3')
    vocals_mp3 = os.path.join(output_dir, 'vocals.mp3')

    # wav → mp3 변환 (이미 있으면 생략)
    if not os.path.exists(accomp_mp3) and os.path.exists(accomp_wav_dst):
        AudioSegment.from_wav(accomp_wav_dst).export(accomp_mp3, format="mp3")
    if not os.path.exists(vocals_mp3) and os.path.exists(vocals_wav_dst):
        AudioSegment.from_wav(vocals_wav_dst).export(vocals_mp3, format="mp3")

    # DB 처리 등 필요시 추가
    data=analyze_audio(accomp_mp3)
    conn = get_conn()
    cur = conn.cursor()
    print(data)
    cur.execute(
        "INSERT INTO music_meta (musicid, pitch_vector, onset_times) VALUES (%s, %s, %s) RETURNING *",
        (musicid, data['pitch_hz'], data['onset_times'])
    )
    cur.execute("INSERT INTO musics (musicid, accompaniment_path) VALUES (%s, %s) RETURNING *",
        (musicid, accomp_mp3))
    
    #cur.execute()
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
    cur.execute("INSERT INTO musics (musicid, accompaniment_path) VALUES (%s, %s) RETURNING *",
        (filename, accomp_mp3))
    
    #cur.execute()
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
        return jsonify({'musicid': updated[0], 'title': updated[1], 'artist': updated[2]}), 200
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
            'SELECT musicid, pitch_vector, onset_times FROM music_meta WHERE musicid = %s',
            (musicid,)
        )
        row = cur.fetchone()
        if not row:
            return jsonify({'error': 'Not found'}), 404
        music_meta = {
            'musicid': row[0],
            'pitch_vector': row[1],   # FLOAT8[] 타입은 psycopg2가 바로 파이썬 리스트로 반환
            'onset_times': row[2]
        }
        return jsonify(music_meta), 200
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
    music_id=data['musicid']

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE musics SET title=%s, artist=%s WHERE musicid=%s RETURNING musicid",
        (title, artist, music_id)
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


    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)