import sys
import os
import tempfile
import uuid
import traceback
import json
import shutil
import soundfile as sf

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDockWidget, 
    QLabel, QFileDialog, QScrollArea, QProgressBar, QMessageBox, 
    QProgressDialog, QMenu, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer, QUrl, QSettings
from PyQt6.QtGui import QAction
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# -- 상대 경로 임포트 --
from ..models import ProjectContext, TextEvent, Track, SourceType, SubtitleItem
from ..audio_engine import AudioEngine
from ..ai_service import (
    AISeparationWorker, SubtitleGenerationWorker, 
    PromptSeparationWorker, DenoiseWorker, ExternalFileLoader
)
from ..media_util import is_video_file

from .visualizer_screen import VisualizerScreen
from .timeline_widget import TimelinePanel
from .track_controls import InspectorPanel, TrackControls
from .detail_settings import SpectrumSettingsWidget, TextSettingsWidget
from .youtube_dialog import YoutubeDialog

class StudioMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Mixing Studio - Final Integrated")
        self.resize(1600, 1000)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: white; }
            QDockWidget { color: white; border: 1px solid #444; }
            QScrollArea { border: none; background-color: #1e1e1e; }
            QProgressBar { text-align: center; color: white; border: 1px solid #555; background: #333; }
            QProgressBar::chunk { background-color: #00aaff; }
            QMenuBar { background-color: #2b2b2b; color: white; }
            QMenuBar::item:selected { background-color: #444; }
            QMenu { background-color: #2b2b2b; color: white; border: 1px solid #555; }
            QMenu::item:selected { background-color: #444; }
        """)

        self.ctx = ProjectContext()
        self.engine = AudioEngine(self.ctx)
        self.selected_track_id = None
        
        self.track_widgets = [] # List of TrackControls
        
        # Visualizer Init
        self.visualizer = VisualizerScreen(self.ctx)
        
        # Video Player Init
        self.video_player = QMediaPlayer(self)
        self.video_audio = QAudioOutput(self)
        self.video_audio.setVolume(0) 
        self.video_player.setAudioOutput(self.video_audio)
        
        if hasattr(self.visualizer, 'video_sink'):
            self.video_player.setVideoOutput(self.visualizer.video_sink)
        
        self.init_ui()
        self.init_menu()
        self.restore_layout()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_playback_ui)
        self.timer.start(33) # ~30fps

    def init_ui(self):
        self.setDockNestingEnabled(True)
        
        # 1. 툴바 (ToolBar)
        tb = self.addToolBar("Main")
        self.b_play = QPushButton("▶ Play"); self.b_play.clicked.connect(self.toggle_play); tb.addWidget(self.b_play)
        b_bg = QPushButton("🖼 BG"); b_bg.clicked.connect(self.set_bg); tb.addWidget(b_bg)
        b_sub = QPushButton("🎤 Subtitles"); b_sub.clicked.connect(self.generate_subtitles); tb.addWidget(b_sub)
        b_tools = QPushButton("🛠 Denoise"); b_tools.clicked.connect(self.run_denoise_on_selected); tb.addWidget(b_tools)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        tb.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel(" Ready"); tb.addWidget(self.lbl_status)
        
        self.visualizer.settingsRequested.connect(self.open_vis_settings)
        self.setCentralWidget(self.visualizer)
        
        # --- Docks ---
        
        # 2. Dock: 타임라인 (하단)
        self.dock_tl = QDockWidget("Timeline", self)
        self.dock_tl.setObjectName("DockTimeline")
        self.tl_panel = TimelinePanel()
        self.tl_panel.view.seek_requested.connect(self.seek)
        
        # 타임라인 시그널 연결
        if hasattr(self.tl_panel.view, 'event_selected'):
            self.tl_panel.view.event_selected.connect(self.on_timeline_event_selected)
        if hasattr(self.tl_panel.view, 'refresh_needed'):
            self.tl_panel.view.refresh_needed.connect(self.refresh_timeline)
        if hasattr(self.tl_panel.view, 'request_settings_signal'):
            self.tl_panel.view.request_settings_signal.connect(self.open_vis_settings)
        
        self.dock_tl.setWidget(self.tl_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_tl)
        
        # 3. Dock: 트랙 리스트 (좌측)
        self.dock_tr = QDockWidget("Layers", self)
        self.dock_tr.setObjectName("DockLayers")
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.cont = QWidget(); 
        self.cont.setStyleSheet("background-color: #1e1e1e;")
        self.lay = QVBoxLayout(self.cont); self.lay.addStretch(1)
        self.scroll.setWidget(self.cont)
        self.dock_tr.setWidget(self.scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_tr)
        
        # 4. Dock: 속성 패널 (우측) - QStackedWidget 적용
        self.dock_ins = QDockWidget("Properties", self)
        self.dock_ins.setObjectName("DockProperties")
        
        self.stack_props = QStackedWidget()
        
        # 4-0. 대기 화면
        self.lbl_prop_empty = QLabel("No Selection")
        self.lbl_prop_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack_props.addWidget(self.lbl_prop_empty)
        
        # 4-1. 트랙 인스펙터
        self.ins = InspectorPanel(self.ctx)
        self.ins.track_modified.connect(self.refresh_timeline)
        # 인스펙터 변경사항을 트랙리스트 UI에 반영
        self.ins.volume_changed.connect(self.sync_track_volume_ui)
        self.ins.pan_changed.connect(self.sync_track_pan_ui)
        self.stack_props.addWidget(self.ins)
        
        # 4-2. 스펙트럼 설정창
        self.prop_spec = SpectrumSettingsWidget()
        self.prop_spec.settings_changed.connect(self.on_spec_settings_changed)
        self.stack_props.addWidget(self.prop_spec)
        
        # 4-3. 텍스트 설정창
        self.prop_text = TextSettingsWidget()
        self.prop_text.settings_changed.connect(self.on_text_settings_changed)
        self.stack_props.addWidget(self.prop_text)
        
        self.dock_ins.setWidget(self.stack_props)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_ins)
        
        self.current_event_item = None

    # ==========================================================
    #  [UI Sync Logic] 트랙 속성 동기화
    # ==========================================================
    def sync_track_volume_ui(self, vol):
        if not self.selected_track_id: return
        widget = next((w for w in self.track_widgets if w.track.id == self.selected_track_id), None)
        if widget:
            widget.blockSignals(True)
            widget.slider_vol.setValue(int(vol * 100))
            widget.blockSignals(False)

    def sync_track_pan_ui(self, pan):
        pass

    # ==========================================================
    #  [Event & Property Selection Logic]
    # ==========================================================
    def on_timeline_event_selected(self, item):
        self.current_event_item = item
        
        if item is None:
            if self.selected_track_id:
                self.stack_props.setCurrentWidget(self.ins)
                self.dock_ins.setWindowTitle("Properties - Track")
            else:
                self.stack_props.setCurrentIndex(0) # Empty
                self.dock_ins.setWindowTitle("Properties")
            return

        if item.p_type == 'subtitle':
            self.stack_props.setCurrentWidget(self.prop_text)
            self.dock_ins.setWindowTitle("Properties - Text")
            if item.sub_item:
                data = {
                    'text': item.sub_item.text,
                    'font_size': item.sub_item.font_size,
                    'color': item.sub_item.color_hex
                }
                self.prop_text.load_settings(data)
            
        elif 'spec' in item.p_type:
            self.stack_props.setCurrentWidget(self.prop_spec)
            self.dock_ins.setWindowTitle("Properties - Spectrum")
            self.prop_spec.load_settings(self.visualizer.get_settings())
        else:
            if self.selected_track_id:
                self.stack_props.setCurrentWidget(self.ins)
            else:
                self.stack_props.setCurrentWidget(self.lbl_prop_empty)

    def on_text_settings_changed(self, data):
        if self.current_event_item and self.current_event_item.sub_item:
            self.current_event_item.sub_item.text = data['text']
            self.current_event_item.sub_item.font_size = data['font_size']
            self.current_event_item.sub_item.color_hex = data['color']
            self.refresh_timeline() 
            self.visualizer.update()

    def on_spec_settings_changed(self, data):
        self.visualizer.update_settings(data)
        self.visualizer.update()

    def open_vis_settings(self):
        self.stack_props.setCurrentWidget(self.prop_spec)
        self.dock_ins.setWindowTitle("Properties - Spectrum")
        self.prop_spec.load_settings(self.visualizer.get_settings())

    # ==========================================================
    #  [기존] 메뉴 및 파일 처리
    # ==========================================================
    def init_menu(self):
        menubar = self.menuBar()
        menubar.clear()
        
        file_menu = menubar.addMenu("파일")
        
        act_open = QAction("불러오기 (영상)...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.load_media_dialog)
        file_menu.addAction(act_open)

        act_load = QAction("불러오기 (.ams 프로젝트)...", self)
        act_load.triggered.connect(self.load_project)
        file_menu.addAction(act_load)
        
        act_save = QAction("저장하기 (.ams 프로젝트)...", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.save_project)
        file_menu.addAction(act_save)
        
        file_menu.addSeparator()
        
        act_yt = QAction("유튜브에서 가져오기...", self)
        act_yt.setShortcut("Ctrl+Y")
        act_yt.triggered.connect(self.open_youtube_dialog)
        file_menu.addAction(act_yt)
        
        file_menu.addSeparator()
        
        act_export = QAction("내보내기 (Mix)...", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self.export_media)
        file_menu.addAction(act_export)

        act_exit = QAction("종료", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

    def load_media_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 파일 열기", "", "Media (*.mp4 *.mkv *.avi *.mov *.mp3 *.wav)")
        if path: self.process_media_file(path)

    def open_youtube_dialog(self):
        dlg = YoutubeDialog(self)
        dlg.videoDownloaded.connect(self._on_youtube_imported)
        dlg.exec()

    def _on_youtube_imported(self, path):
        self.process_media_file(path)

    def process_media_file(self, path):
        safe_path = os.path.abspath(path)
        if not os.path.exists(safe_path):
            QMessageBox.critical(self, "Error", f"파일을 찾을 수 없습니다:\n{safe_path}")
            return

        self.engine.stop()
        self.video_player.stop()
        self.ctx.clear()
        self.b_play.setText("▶ Play")
        
        for w in self.track_widgets: w.deleteLater()
        self.track_widgets = []
        self.ins.set_track(None)
        self.visualizer.clear_background()
        self.tl_panel.refresh(self.ctx)
        
        if is_video_file(safe_path): 
            self.ctx.video_path = safe_path 
            self.video_player.setSource(QUrl.fromLocalFile(safe_path))
            self.video_player.play()
            QTimer.singleShot(300, self.video_player.pause)
        else: 
            self.video_player.setSource(QUrl())
            
        self.lbl_status.setText("AI 레이어 분리 중... (잠시만 기다려주세요)")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.worker = AISeparationWorker(safe_path, user_sr=44100, device_pref="cuda")
        self.worker.finished.connect(self.on_ready)
        if hasattr(self.worker, 'progress'): 
            self.worker.progress.connect(self.update_progress)
        self.worker.start()

    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(msg)

    def on_ready(self, tracks):
        self.lbl_status.setText("분리 완료")
        self.progress_bar.setVisible(False)
        for t in tracks: self.add_track_to_ui(t)
        self.refresh_timeline()
        QMessageBox.information(self, "성공", f"총 {len(tracks)}개의 레이어로 분리되었습니다.")

    def add_track_to_ui(self, t: Track):
        if t not in self.ctx.tracks: self.ctx.add_track(t)
        
        w = TrackControls(t)
        if hasattr(w, 'clicked'):
            w.clicked.connect(self.sel_track)
        if hasattr(w, 'separationRequested'):
            w.separationRequested.connect(self.start_prompt_separation)
        
        w.volume_changed.connect(lambda v: self.engine.set_track_volume(t.name, v))
        w.volume_changed.connect(self.sync_inspector_volume) 
        w.mute_toggled.connect(lambda m: self.engine.set_track_mute(t.name, m))
        w.solo_toggled.connect(lambda s: self.engine.set_track_solo(t.name, s))
            
        self.lay.insertWidget(self.lay.count()-1, w)
        self.track_widgets.append(w)
        w.show()

    def sync_inspector_volume(self, vol):
        if self.ins.current_track and self.selected_track_id == self.ins.current_track.id:
            self.ins.blockSignals(True)
            self.ins.slider_vol.setValue(int(vol * 100))
            self.ins.blockSignals(False)

    def sel_track(self, tid):
        self.selected_track_id = tid
        t = next((x for x in self.ctx.tracks if x.id == tid), None)
        
        self.stack_props.setCurrentWidget(self.ins)
        self.dock_ins.setWindowTitle("Properties - Track")
        self.ins.set_track(t)
        
        for w in self.track_widgets: 
            if hasattr(w, 'set_selected'): w.set_selected(w.track.id == tid)

    def refresh_timeline(self):
        self.tl_panel.refresh(self.ctx)

    def set_bg(self):
        p, _ = QFileDialog.getOpenFileName(self, "BG", "", "Img (*.png *.jpg *.jpeg)")
        if p: self.visualizer.set_background_image(p)

    # ==========================================================
    #  [Playback & Automation]
    # ==========================================================
    def toggle_play(self):
        if self.ctx.is_playing:
            self.engine.stop(); self.video_player.pause(); self.b_play.setText("▶ Play")
        else:
            if self.ctx.total_frames > 0:
                if self.ctx.current_frame >= self.ctx.total_frames:
                    self.seek(0)
                self.engine.start(); self.video_player.play(); self.b_play.setText("⏸ Pause")

    def seek(self, sec):
        frame = int(sec * self.ctx.sample_rate)
        self.engine.seek(frame)
        if self.video_player.source().isValid(): self.video_player.setPosition(int(sec*1000))
        self.update_playback_ui()

    def update_playback_ui(self):
        # 1. 자동 정지 체크
        if self.ctx.is_playing and self.ctx.total_frames > 0:
            if self.ctx.current_frame >= self.ctx.total_frames:
                self.engine.stop()
                self.video_player.pause()
                self.b_play.setText("▶ Play")
                self.ctx.current_frame = self.ctx.total_frames
                
        cur_sec = self.ctx.current_frame / max(1, self.ctx.sample_rate)
        
        # 2. 오디오 버퍼 시각화
        data = self.engine.vis_buffer if self.ctx.is_playing else None
        self.visualizer.set_audio_data(data)
        
        # 3. [FIX] 오토메이션 (스펙트럼 설정)
        # 타임라인 값이 없으면(-1.0) 자동으로 끄도록(False) 처리
        vis_val = self.ctx.get_global_value('spec_visible', cur_sec, -1.0)
        if vis_val >= 0:
            self.visualizer.settings['spec_visible'] = (vis_val >= 0.5)
        else:
            # 이벤트 구간 밖이면 무조건 끔
            self.visualizer.settings['spec_visible'] = False
            
        scale_val = self.ctx.get_global_value('spec_scale', cur_sec, -1.0) 
        if scale_val >= 0:
            self.visualizer.settings['height_scale'] = scale_val
            
        shape_val = self.ctx.get_global_value('spec_shape', cur_sec, -1)
        if shape_val >= 0:
            modes = ['linear', 'circular', 'line']
            idx = int(shape_val) % 3
            self.visualizer.settings['shape'] = modes[idx]
        
        # 4. 자막 동기화
        active_text = ""
        for sub in self.ctx.subtitles:
            if sub.start_time <= cur_sec <= sub.end_time:
                active_text = sub.text; break 
        self.visualizer.sub_cfg['text'] = active_text
        
        self.visualizer.update()

        # 5. 타임라인 & 트랙 위젯 업데이트
        if self.ctx.total_frames > 0:
            self.tl_panel.update_playhead(cur_sec)
            ratio = self.ctx.current_frame / self.ctx.total_frames
            for w in self.track_widgets: 
                if hasattr(w, 'set_playhead_ratio'): w.set_playhead_ratio(ratio)
            
        # 6. 비디오 싱크
        if self.ctx.is_playing and self.video_player.source().isValid():
            v = self.video_player.position(); a = int(cur_sec * 1000)
            if abs(v-a) > 150: self.video_player.setPosition(a)
            if self.video_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState: 
                self.video_player.play()

    # ==========================================================
    #  [AI & Project Tools]
    # ==========================================================
    def generate_subtitles(self):
        target = None
        if self.selected_track_id:
            target = next((t for t in self.ctx.tracks if t.id == self.selected_track_id), None)
        if not target:
            target = next((t for t in self.ctx.tracks if "Vocals" in t.name or "vocals" in t.name), None)
            
        if not target:
            QMessageBox.critical(self, "Error", "Please select a Vocal track.")
            return

        tmp = os.path.join(tempfile.gettempdir(), f"temp_{uuid.uuid4()}.wav")
        try:
            sf.write(tmp, target.data, target.sr)
        except Exception as e:
            print(e); return

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
        self.refresh_timeline()
        QMessageBox.information(self, "Success", "Subtitles Generated!")

    def run_denoise_on_selected(self):
        if not self.selected_track_id:
            QMessageBox.warning(self, "Warning", "Select a track first!")
            return
        target = next((t for t in self.ctx.tracks if t.id == self.selected_track_id), None)
        if not target: return
        
        self.denoise_progress = QProgressDialog("AI 노이즈 제거 중...", "취소", 0, 100, self)
        self.denoise_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.denoise_progress.setAutoClose(False)
        self.denoise_progress.show()

        self.denoise_worker = DenoiseWorker(target, amount=0.9)
        self.denoise_worker.progress.connect(self.denoise_progress.setValue)
        self.denoise_worker.progress.connect(lambda v, m: self.denoise_progress.setLabelText(m) if isinstance(m, str) else None)
        self.denoise_worker.finished.connect(self.on_denoise_finished)
        self.denoise_worker.failed.connect(self.on_worker_failed)
        self.denoise_worker.start()

    def on_denoise_finished(self, new_track):
        self.denoise_progress.close()
        self.add_track_to_ui(new_track)
        self.refresh_timeline()
        QMessageBox.information(self, "완료", "노이즈 제거 완료")
        
    def start_prompt_separation(self, track_id: str, prompt: str):
        target_track = next((t for t in self.ctx.tracks if t.id == track_id), None)
        if not target_track: return

        self.prompt_progress = QProgressDialog("AI(Prompt) 처리 중...", "취소", 0, 100, self)
        self.prompt_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.prompt_progress.setAutoClose(False)
        self.prompt_progress.show()

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

    def on_prompt_separation_finished(self, new_tracks: list):
        self.prompt_progress.close()
        if not new_tracks: return
        for t in new_tracks:
            self.add_track_to_ui(t)
        self.refresh_timeline()
        QMessageBox.information(self, "완료", f"'{new_tracks[0].name}' 추출이 완료되었습니다.")

    def on_worker_failed(self, err_msg: str):
        if hasattr(self, 'prompt_progress'): self.prompt_progress.close()
        if hasattr(self, 'denoise_progress'): self.denoise_progress.close()
        QMessageBox.critical(self, "오류", f"작업 실패: {err_msg}")

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "AMS Project (*.ams)")
        if not path: return
        
        project_dir = os.path.splitext(path)[0] + "_data"
        if os.path.exists(project_dir): shutil.rmtree(project_dir)
        os.makedirs(project_dir, exist_ok=True)
        
        tracks_meta = []
        for i, t in enumerate(self.ctx.tracks):
            wav_name = f"track_{i}_{t.id}.wav"
            wav_path = os.path.join(project_dir, wav_name)
            sf.write(wav_path, t.data, t.sr)
            
            meta = t.to_dict()
            meta['data_file'] = wav_name
            tracks_meta.append(meta)
        
        if self.ctx.video_path and os.path.exists(self.ctx.video_path):
            vid_name = "video_" + os.path.basename(self.ctx.video_path)
            dest_vid = os.path.join(project_dir, vid_name)
            try:
                shutil.copy2(self.ctx.video_path, dest_vid)
                self.ctx.video_path = dest_vid
            except: pass
            
        proj_data = self.ctx.to_dict()
        proj_data['tracks'] = tracks_meta
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(proj_data, f, indent=2)
            
        self.lbl_status.setText(f"Project Saved: {os.path.basename(path)}")
        QMessageBox.information(self, "Saved", "Project saved successfully.")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "AMS Project (*.ams)")
        if not path: return
        
        project_dir = os.path.splitext(path)[0] + "_data"
        if not os.path.exists(project_dir):
            QMessageBox.critical(self, "Error", "Data folder not found!")
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.engine.stop()
            self.video_player.stop()
            self.ctx.clear()
            for w in self.track_widgets: w.deleteLater()
            self.track_widgets = []
            
            self.ctx.load_from_dict(data)
            
            if self.ctx.video_path:
                if os.path.exists(self.ctx.video_path):
                    self.video_player.setSource(QUrl.fromLocalFile(self.ctx.video_path))
                    self.video_player.play()
                    QTimer.singleShot(300, self.video_player.pause)
                else:
                    local_vid = os.path.join(project_dir, os.path.basename(self.ctx.video_path))
                    if os.path.exists(local_vid):
                        self.ctx.video_path = local_vid
                        self.video_player.setSource(QUrl.fromLocalFile(local_vid))
                        self.video_player.play()
                        QTimer.singleShot(300, self.video_player.pause)
            
            for t_meta in data.get('tracks', []):
                wav_path = os.path.join(project_dir, t_meta['data_file'])
                if os.path.exists(wav_path):
                    audio_data, file_sr = sf.read(wav_path, dtype='float32', always_2d=True)
                    t = Track.from_dict(t_meta, audio_data)
                    self.add_track_to_ui(t)
            
            self.refresh_timeline()
            self.lbl_status.setText(f"Project Loaded: {os.path.basename(path)}")
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Load failed: {e}")

    def export_media(self):
        if not self.ctx.tracks: return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "mix.wav", "Audio Only (*.wav)")
        if not path: return
        if self.engine.export_mix_to_wav(path): QMessageBox.information(self, "Success", f"Exported: {path}")
        else: QMessageBox.critical(self, "Error", "Export Failed")

    def closeEvent(self, e):
        settings = QSettings("Arscabar", "AiMixingStudio")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        self.engine.stop()
        super().closeEvent(e)

    def restore_layout(self):
        settings = QSettings("Arscabar", "AiMixingStudio")
        if settings.value("geometry"):
            self.restoreGeometry(settings.value("geometry"))
        if settings.value("windowState"):
            self.restoreState(settings.value("windowState"))