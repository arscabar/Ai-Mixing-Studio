import os
import shutil
import uuid
import time
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLineEdit, QLabel, QProgressBar, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, QUrl, Qt
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView

import yt_dlp

# [설정] 임시 다운로드 폴더 이름
TEMP_FOLDER_NAME = "temp_downloads"
# [설정] 며칠 지난 파일을 지울 것인가? (3일)
DELETE_AFTER_DAYS = 3

# 로그 숨기기
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--log-level=3"
os.environ["QT_LOGGING_RULES"] = "qt.webenginecontext.debug=false;qt.multimedia.ffmpeg.debug=false"

class NoLogWebPage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, msg, line, sourceID):
        pass

class YoutubeDownloadWorker(QThread):
    progress_msg = pyqtSignal(str)
    progress_pct = pyqtSignal(int)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, url):
        super().__init__()
        self.url = url
        
        # [수정] 프로젝트 루트 경로에 임시 폴더 생성
        self.base_dir = os.getcwd() # 현재 프로그램 실행 위치
        self.temp_dir = os.path.join(self.base_dir, TEMP_FOLDER_NAME)
        
        # 폴더가 없으면 생성
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def my_hook(self, d):
        if d['status'] == 'downloading':
            try:
                p = d.get('_percent_str', '0').replace('%', '')
                self.progress_pct.emit(int(float(p)))
                self.progress_msg.emit(f"다운로드 중... {p}%")
            except:
                pass
        elif d['status'] == 'finished':
            self.progress_pct.emit(100)
            self.progress_msg.emit("파일 변환 및 최적화 중... (잠시만 기다려주세요)")

    def get_ffmpeg_path(self):
        path = r"E:\ffmpeg\bin"
        if not os.path.exists(path):
            exe = shutil.which("ffmpeg")
            if exe: path = os.path.dirname(exe)
            else: return None
        return path

    def run(self):
        ffmpeg_path = self.get_ffmpeg_path()
        if not ffmpeg_path:
            self.failed.emit("FFmpeg를 찾을 수 없습니다.")
            return

        # 파일명 중복 방지 (랜덤 ID)
        random_id = str(uuid.uuid4())[:8]
        
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best', 
            'merge_output_format': 'mp4',
            'ffmpeg_location': ffmpeg_path,
            
            # [수정] 지정된 임시 폴더에 저장
            'outtmpl': os.path.join(self.temp_dir, f'%(title)s_{random_id}.%(ext)s'),
            
            'progress_hooks': [self.my_hook],
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            
            # 강력 재인코딩 (H.264)
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'postprocessor_args': {
                'merger': ['-c:v', 'libx264', '-c:a', 'aac', '-pix_fmt', 'yuv420p'],
                'videoconvertor': ['-c:v', 'libx264', '-c:a', 'aac', '-pix_fmt', 'yuv420p']
            }
        }

        try:
            self.progress_msg.emit("다운로드 시작...")
            
            target_filename = ""
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                target_filename = ydl.prepare_filename(info)
                
                base, ext = os.path.splitext(target_filename)
                if ext != '.mp4':
                    target_filename = base + ".mp4"

            if not os.path.exists(target_filename):
                 if os.path.exists(base + ".mp4"):
                     target_filename = base + ".mp4"
                 else:
                    raise FileNotFoundError("변환된 파일을 찾을 수 없습니다.")

            self.finished.emit(target_filename)
            
        except Exception as e:
            self.failed.emit(str(e))

class YoutubeDialog(QDialog):
    videoDownloaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YouTube Video Downloader")
        self.resize(1000, 700)
        
        # [기능 추가] 창 열 때 오래된 임시 파일 청소
        self.cleanup_temp_files()

        layout = QVBoxLayout(self)
        
        # Top Bar
        top_bar = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("YouTube URL 입력")
        self.btn_go = QPushButton("이동")
        self.btn_download = QPushButton("⬇ 다운로드 (H.264 변환)")
        self.btn_download.setStyleSheet("background-color: #cc0000; color: white; font-weight: bold;")
        
        top_bar.addWidget(QLabel("URL:"))
        top_bar.addWidget(self.url_input)
        top_bar.addWidget(self.btn_go)
        top_bar.addWidget(self.btn_download)
        layout.addLayout(top_bar)
        
        # Browser
        self.webview = QWebEngineView()
        self.custom_page = NoLogWebPage(self.webview)
        self.webview.setPage(self.custom_page)
        self.webview.setUrl(QUrl("https://www.youtube.com"))
        self.webview.urlChanged.connect(self._on_url_changed)
        layout.addWidget(self.webview)
        
        # Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("font-weight: bold; color: yellow;")
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.lbl_status)
        
        # Connections
        self.btn_go.clicked.connect(lambda: self.webview.setUrl(QUrl(self.url_input.text())))
        self.btn_download.clicked.connect(self.start_download)

    def _on_url_changed(self, url):
        self.url_input.setText(url.toString())

    def closeEvent(self, event):
        self.webview.setUrl(QUrl("about:blank"))
        super().closeEvent(event)

    def cleanup_temp_files(self):
        """설정된 기간(3일)보다 오래된 임시 파일 삭제"""
        temp_dir = os.path.join(os.getcwd(), TEMP_FOLDER_NAME)
        if not os.path.exists(temp_dir):
            return

        now = time.time()
        cutoff = DELETE_AFTER_DAYS * 86400 # 일 * 초
        deleted_count = 0

        try:
            for f in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, f)
                # 파일이고, 확장자가 mp4/webm/mkv 등인 경우
                if os.path.isfile(file_path):
                    # 파일 수정 시간 확인
                    file_mtime = os.stat(file_path).st_mtime
                    if file_mtime < (now - cutoff):
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except PermissionError:
                            # 사용 중인 파일은 패스
                            pass
            
            if deleted_count > 0:
                print(f"[알림] 오래된 임시 파일 {deleted_count}개를 삭제했습니다.")
                
        except Exception as e:
            print(f"[경고] 임시 파일 정리 중 오류: {e}")

    def start_download(self):
        url = self.url_input.text()
        if "youtube.com/watch" not in url and "youtu.be" not in url:
            QMessageBox.warning(self, "Invalid URL", "유효한 YouTube 동영상 페이지가 아닙니다.")
            return

        self.btn_download.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("준비 중...")
        
        self.worker = YoutubeDownloadWorker(url)
        self.worker.progress_msg.connect(self.lbl_status.setText)
        self.worker.progress_pct.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, path):
        self.progress_bar.setValue(100)
        self.lbl_status.setText("완료!")
        self.btn_download.setEnabled(True)
        self.webview.setUrl(QUrl("about:blank"))
        self.videoDownloaded.emit(path)
        self.accept()

    def _on_failed(self, err):
        self.progress_bar.setVisible(False)
        self.btn_download.setEnabled(True)
        self.lbl_status.setText(f"오류 발생")
        QMessageBox.critical(self, "Download Error", str(err))