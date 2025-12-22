# src/ui/main_window.py
import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, QDockWidget, QLabel, QFileDialog, QScrollArea, QProgressBar)
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from ..models import ProjectContext, TextEvent
from ..audio_engine import AudioEngine
from ..ai_service import AISeparationWorker
from ..media_util import is_video_file

from .visualizer_screen import VisualizerScreen
from .timeline_widget import TimelinePanel
from .track_controls import InspectorPanel, TrackStrip

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
        
        self.track_widgets = []
        self._init_ui()
        
        self.video_player = QMediaPlayer(self)
        self.video_audio = QAudioOutput(self)
        self.video_audio.setVolume(0)
        self.video_player.setAudioOutput(self.video_audio)
        self.video_player.setVideoOutput(self.visualizer.video_sink)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_playback_ui)
        self.timer.start(33)

    def _init_ui(self):
        self.setDockNestingEnabled(True)
        tb = self.addToolBar("Main")
        
        b_load = QPushButton("📂 Load"); b_load.clicked.connect(self.load_media); tb.addWidget(b_load)
        self.b_play = QPushButton("▶ Play"); self.b_play.clicked.connect(self.toggle_play); tb.addWidget(self.b_play)
        b_bg = QPushButton("🖼 BG"); b_bg.clicked.connect(self.set_bg); tb.addWidget(b_bg)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        tb.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel(" Ready"); tb.addWidget(self.lbl_status)
        
        self.visualizer = VisualizerScreen()
        self.visualizer.settingsRequested.connect(self.open_vis_settings)
        self.setCentralWidget(self.visualizer)
        
        self.dock_tl = QDockWidget("Timeline", self)
        self.tl_panel = TimelinePanel()
        self.tl_panel.view.seek_requested.connect(self.seek)
        self.tl_panel.view.refresh_needed.connect(self.refresh_timeline)
        # [NEW] 설정창 요청 시그널 연결
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
        
        if dlg.exec(): 
            self.visualizer.set_settings(dlg.get_settings())

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
        
        # Apply Global Automation
        spec_scale = self.ctx.get_global_value('spectrum_scale', cur_sec, 0.1) 
        self.visualizer.scale = spec_scale
        
        # Apply Text
        active_text = ""
        for evt in self.ctx.text_events:
            if evt.is_persistent or (evt.start_time <= cur_sec <= evt.end_time):
                active_text = evt.text
                break 
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
        if hasattr(self.worker, 'progress'):
            self.worker.progress.connect(self.update_progress)
        self.worker.start()

    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(msg)

    def on_ready(self, tracks):
        self.lbl_status.setText("Ready")
        self.progress_bar.setVisible(False)
        for t in tracks:
            self.ctx.add_track(t)
            w = TrackStrip(t); w.clicked.connect(self.sel_track)
            self.lay.insertWidget(self.lay.count()-1, w)
            self.track_widgets.append(w)
            w.show()
        self.refresh_timeline()

    def sel_track(self, tid):
        t = next((x for x in self.ctx.tracks if x.id == tid), None)
        self.ins.set_track(t)
        for w in self.track_widgets: w.set_selected(w.track.id == tid)

    def refresh_timeline(self):
        self.tl_panel.refresh(self.ctx)

    def set_bg(self):
        p, _ = QFileDialog.getOpenFileName(self, "BG", "", "Img (*.png *.jpg)")
        if p: self.visualizer.set_background_image(p)

    def closeEvent(self, e):
        self.engine.stop(); super().closeEvent(e)