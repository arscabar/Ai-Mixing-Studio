# src/ffmpeg_util.py
import os
import subprocess
import shutil

def find_ffmpeg() -> str:
    # 1. Check local bin folder
    local_bin = os.path.join(os.getcwd(), "bin", "ffmpeg.exe")
    if os.path.exists(local_bin):
        return local_bin
    
    # 2. Check system PATH
    sys_path = shutil.which("ffmpeg")
    if sys_path:
        return sys_path
        
    return None

def extract_audio_to_wav(video_path: str, out_wav_path: str):
    ffmpeg_exe = find_ffmpeg()
    if not ffmpeg_exe:
        raise FileNotFoundError("FFmpeg not found. Please install FFmpeg or place ffmpeg.exe in /bin folder.")

    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-vn", 
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        out_wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def merge_audio_to_video(video_source: str, audio_source: str, out_path: str):
    ffmpeg_exe = find_ffmpeg()
    if not ffmpeg_exe:
        raise FileNotFoundError("FFmpeg not found.")

    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_source,
        "-i", audio_source,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)