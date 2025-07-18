
from flask import Flask, request, send_file, jsonify
import os
from spleeter.separator import Separator
from werkzeug.utils import secure_filename
from pydub import AudioSegment

app=Flask(__name__)
app.debug=True

@app.route("/")
def index():
    return "hello"

app = Flask(__name__)
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


@app.route('/rank', methods=['GET'])
def rank_show():
    return "hello"
    
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)