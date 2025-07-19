import yt_dlp

def download_audio_with_ytdlp(youtube_url, filename):
    ydl_opts = {
        'format': 'bestaudio/best',  # 최고 품질의 오디오 형식 지정
        'extractaudio': True,  # 오디오만 추출
        'audioformat': 'wav',  # 오디오 형식을 wav로 지정
        'outtmpl': f'{filename}.%(ext)s',  # 파일 이름 지정 및 확장자 설정
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

# 예시 사용법
youtube_url = 'https://www.youtube.com/watch?v=13-YekiSLQ4'
download_audio_with_ytdlp(youtube_url, 'audio')  # 오디오 다운로드