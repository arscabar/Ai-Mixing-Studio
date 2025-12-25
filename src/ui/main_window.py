# src/ui/main_window.py
import sys
import os
import tempfile
import uuid
import traceback
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QDockWidget, 
    QLabel, QFileDialog, QScrollArea, QProgressBar, QMessageBox, 
    QProgressDialog
)
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from ..models import ProjectContext, TextEvent, Track
from ..audio_engine import AudioEngine
# [MODIFIED] PromptSeparationWorker 추가
from ..ai_service import AISeparationWorker, SubtitleGenerationWorker, PromptSeparationWorker
from ..media_util import is_video_file

from .visualizer_screen import VisualizerScreen
from .timeline_widget import TimelinePanel
# [MODIFIED] TrackStrip -> TrackControls (이전 단계의 클래스명 반영)
# 만약 track_controls.py에 TrackStrip 클래스가 있다면 이름을 맞춰주세요.
from .track_controls import InspectorPanel, TrackControls 

class StudioMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Mixing Studio - Final Fixed")
        self.resize(1600, 1000)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: white; }
            QDockWidget { color: white; border: 1px solid #444; }
            QScrollArea { border: none; background-color: #1e1e1e; }
            QProgressBar { text-align: center; color: white; border: 1px solid #555; background: #333; }
            QProgressBar::chunk { background-color: #00aaff; }
        """)

        self.ctx = ProjectContext()
        self.engine = AudioEngine(self.ctx)
        self.selected_track_id = None
        
        self.track_widgets = []
        
        # Pass Context to Visualizer
        self.visualizer = VisualizerScreen(self.ctx)
        
        self.video_player = QMediaPlayer(self)
        self.video_audio = QAudioOutput(self)
        self.video_audio.setVolume(0)
        self.video_player.setAudioOutput(self.video_audio)
        self.video_player.setVideoOutput(self.visualizer.video_sink)
        
        self._init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_playback_ui)
        self.timer.start(33)

    def _init_ui(self):
        self.setDockNestingEnabled(True)
        tb = self.addToolBar("Main")
        
        b_load = QPushButton("📂 Load"); b_load.clicked.connect(self.load_media); tb.addWidget(b_load)
        self.b_play = QPushButton("▶ Play"); self.b_play.clicked.connect(self.toggle_play); tb.addWidget(self.b_play)
        b_bg = QPushButton("🖼 BG"); b_bg.clicked.connect(self.set_bg); tb.addWidget(b_bg)
        
        # Subtitle Button
        b_sub = QPushButton("🎤 Subtitles"); b_sub.clicked.connect(self.generate_subtitles); tb.addWidget(b_sub)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        tb.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel(" Ready"); tb.addWidget(self.lbl_status)
        
        self.visualizer.settingsRequested.connect(self.open_vis_settings)
        self.setCentralWidget(self.visualizer)
        
        self.dock_tl = QDockWidget("Timeline", self)
        self.tl_panel = TimelinePanel()
        self.tl_panel.view.seek_requested.connect(self.seek)
        self.tl_panel.view.refresh_needed.connect(self.refresh_timeline)
        self.tl_panel.view.request_settings_signal.connect(self.open_vis_settings)
        
        self.dock_tl.setWidget(self.tl_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_tl)
        
        self.dock_tr = QDockWidget("Layers", self)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.cont = QWidget(); 
        self.cont.setStyleSheet("background-color: #1e1e1e;")
        self.lay = QVBoxLayout(self.cont); self.lay.addStretch(1)
        self.scroll.setWidget(self.cont)
        self.dock_tr.setWidget(self.scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_tr)
        
        self.dock_ins = QDockWidget("Inspector", self)
        self.ins = InspectorPanel(self.ctx)
        self.ins.track_modified.connect(self.refresh_timeline)
        self.dock_ins.setWidget(self.ins)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_ins)

    def open_vis_settings(self):
        from .visualizer_screen import VisualizerSettingsDialog
        dlg = VisualizerSettingsDialog(self, self.visualizer.get_settings())
        dlg.textAdded.connect(self._on_text_added)
        if dlg.exec(): self.visualizer.set_settings(dlg.get_settings())

    def _on_text_added(self, text, duration):
        t = self.ctx.current_frame / self.ctx.sample_rate
        self.ctx.text_events.append(TextEvent(text=text, start_time=t, end_time=t+duration))
        self.refresh_timeline()
        self.visualizer.update()

    def toggle_play(self):
        if self.ctx.is_playing:
            self.engine.stop(); self.video_player.pause(); self.b_play.setText("▶ Play")
        else:
            if self.ctx.total_frames > 0:
                self.engine.start(); self.video_player.play(); self.b_play.setText("⏸ Pause")

    def seek(self, sec):
        frame = int(sec * self.ctx.sample_rate)
        self.engine.seek(frame)
        if self.video_player.source().isValid(): self.video_player.setPosition(int(sec*1000))
        self.update_playback_ui()

    def update_playback_ui(self):
        cur_sec = self.ctx.current_frame / max(1, self.ctx.sample_rate)
        
        data = self.engine.vis_buffer if self.ctx.is_playing else None
        self.visualizer.set_audio_data(data)
        
        # Apply Global Automations (Visible, Scale, Shape)
        vis_val = self.ctx.get_global_value('spec_visible', cur_sec, 1.0)
        self.visualizer.spec_visible = (vis_val >= 0.5)
        
        scale_val = self.ctx.get_global_value('spec_scale', cur_sec, 1.0) 
        self.visualizer.spec_scale_factor = scale_val
        
        shape_val = self.ctx.get_global_value('spec_shape', cur_sec, -1)
        self.visualizer.spec_shape_override = shape_val
        
        active_text = ""
        for evt in self.ctx.text_events:
            if evt.is_persistent or (evt.start_time <= cur_sec <= evt.end_time):
                active_text = evt.text; break 
        self.visualizer.sub_cfg['text'] = active_text
        self.visualizer.update()

        if self.ctx.total_frames > 0:
            self.tl_panel.update_playhead(cur_sec)
            ratio = self.ctx.current_frame / self.ctx.total_frames
            for w in self.track_widgets: w.set_playhead_ratio(ratio)
            
        if self.ctx.is_playing and self.video_player.source().isValid():
            v = self.video_player.position(); a = int(cur_sec * 1000)
            if abs(v-a) > 150: self.video_player.setPosition(a)
            if self.video_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState: self.video_player.play()

    def load_media(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load", "", "Media (*.mp3 *.wav *.mp4)")
        if not path: return
        self.engine.stop(); self.video_player.stop(); self.ctx.clear()
        
        for w in self.track_widgets: w.deleteLater()
        self.track_widgets = []
        self.ins.set_track(None)
        self.visualizer.clear_background()
        self.tl_panel.refresh(self.ctx)
        
        if is_video_file(path): self.video_player.setSource(QUrl.fromLocalFile(path))
        else: self.video_player.setSource(QUrl())
            
        self.lbl_status.setText("Separating...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.worker = AISeparationWorker(path, user_sr=44100, device_pref="cuda")
        self.worker.finished.connect(self.on_ready)
        if hasattr(self.worker, 'progress'): self.worker.progress.connect(self.update_progress)
        self.worker.start()

    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(msg)

    # [MODIFIED] 트랙 추가 로직을 add_track_to_ui로 분리
    def on_ready(self, tracks):
        self.lbl_status.setText("Ready")
        self.progress_bar.setVisible(False)
        for t in tracks:
            self.add_track_to_ui(t)
        self.refresh_timeline()

    # [NEW] 공통 트랙 추가 메서드
    def add_track_to_ui(self, t: Track):
        # 1. Context에 추가 (이미 추가된 상태일 수 있으므로 체크하거나, 
        #    호출하는 쪽에서 add_track을 하지 않았다면 여기서 수행)
        if t not in self.ctx.tracks:
            self.ctx.add_track(t)
            
        # 2. UI 위젯 생성 (TrackControls 사용)
        w = TrackControls(t)
        
        # 3. 시그널 연결
        # (주의: TrackControls에 clicked 시그널이 없다면 구현해야 함. 
        #  여기서는 클릭 시 선택 로직을 위해 임의로 clicked 연결을 시도하거나 
        #  mousePressEvent 등에서 처리가 필요함)
        if hasattr(w, 'clicked'):
            w.clicked.connect(self.sel_track)
            
        # [NEW] 프롬프트 분리 요청 시그널 연결
        if hasattr(w, 'separationRequested'):
            w.separationRequested.connect(self.start_prompt_separation)
            
        self.lay.insertWidget(self.lay.count()-1, w)
        self.track_widgets.append(w)
        w.show()

    def sel_track(self, tid):
        self.selected_track_id = tid
        t = next((x for x in self.ctx.tracks if x.id == tid), None)
        self.ins.set_track(t)
        for w in self.track_widgets: w.set_selected(w.track.id == tid)

    def refresh_timeline(self):
        self.tl_panel.refresh(self.ctx)

    def set_bg(self):
        p, _ = QFileDialog.getOpenFileName(self, "BG", "", "Img (*.png *.jpg)")
        if p: self.visualizer.set_background_image(p)

    def generate_subtitles(self):
        # 1. Select Vocal
        target = None
        if self.selected_track_id:
            target = next((t for t in self.ctx.tracks if t.id == self.selected_track_id), None)
        if not target:
            target = next((t for t in self.ctx.tracks if "Vocals" in t.name or "vocals" in t.name), None)
            
        if not target:
            QMessageBox.critical(self, "Error", "Please select a Vocal track.")
            return

        # 2. Temp File
        tmp = os.path.join(tempfile.gettempdir(), f"temp_{uuid.uuid4()}.wav")
        try:
            import soundfile as sf
            sf.write(tmp, target.data, target.sr)
        except Exception as e:
            print(e); return

        # 3. Worker
        self.sub_worker = SubtitleGenerationWorker(tmp, model_size="small")
        self.sub_worker.progress.connect(self.update_progress)
        self.sub_worker.finished.connect(self.on_sub_ready)
        self.progress_bar.setVisible(True)
        self.lbl_status.setText("Whisper Running...")
        self.sub_worker.start()

    def on_sub_ready(self, subs):
        self.ctx.subtitles = subs
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"Subtitles: {len(subs)}")
        QMessageBox.information(self, "Success", "Subtitles Generated!")

    # [NEW] 프롬프트 분리 시작 메서드
    def start_prompt_separation(self, track_id: str, prompt: str):
        target_track = next((t for t in self.ctx.tracks if t.id == track_id), None)
        if not target_track: return

        # 모달 다이얼로그 생성
        self.prompt_progress = QProgressDialog("AI(Prompt) 처리 중...", "취소", 0, 100, self)
        self.prompt_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.prompt_progress.setAutoClose(False)
        self.prompt_progress.show()

        # 워커 설정
        self.prompt_worker = PromptSeparationWorker(
            input_track_data=target_track.data,
            sr=target_track.sr,
            prompt=prompt,
            device_pref="cuda"
        )
        self.prompt_worker.progress.connect(self.prompt_progress.setValue)
        self.prompt_worker.progress.connect(lambda v, m: self.prompt_progress.setLabelText(m) if isinstance(m, str) else None)
        self.prompt_worker.finished.connect(self.on_prompt_separation_finished)
        self.prompt_worker.failed.connect(self.on_worker_failed)
        
        self.prompt_progress.canceled.connect(self.prompt_worker.requestInterruption)
        self.prompt_worker.start()

    # [NEW] 프롬프트 분리 완료 핸들러
    def on_prompt_separation_finished(self, new_tracks: list):
        self.prompt_progress.close()
        if not new_tracks: return
        
        # 새 트랙 UI 추가
        for t in new_tracks:
            self.add_track_to_ui(t)
            
        self.refresh_timeline()
        QMessageBox.information(self, "완료", f"'{new_tracks[0].name}' 추출이 완료되었습니다.")

    # [NEW] 작업 실패 핸들러
    def on_worker_failed(self, err_msg: str):
        if hasattr(self, 'prompt_progress'):
            self.prompt_progress.close()
        QMessageBox.critical(self, "오류", f"작업 실패: {err_msg}")

    def export_media(self):
        if not self.ctx.tracks: return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "mix.wav", "Audio Only (*.wav)")
        if not path: return
        if self.engine.export_mix_to_wav(path): QMessageBox.information(self, "Success", f"Exported: {path}")
        else: QMessageBox.critical(self, "Error", "Export Failed")

    def closeEvent(self, e):
        self.engine.stop(); super().closeEvent(e)