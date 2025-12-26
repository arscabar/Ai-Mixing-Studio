from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QPushButton, QFrame, QInputDialog,
                             QDial, QLineEdit, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen

class TrackControls(QWidget):
    """
    좌측 트랙 리스트 위젯
    """
    clicked = pyqtSignal(str) # track_id
    separationRequested = pyqtSignal(str, str) # track_id, prompt
    
    # AudioEngine 연동 시그널
    volume_changed = pyqtSignal(float)
    mute_toggled = pyqtSignal(bool)
    solo_toggled = pyqtSignal(bool)

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.playhead_ratio = 0.0
        self.init_ui()
        self.setMouseTracking(True)

    def init_ui(self):
        self.setFixedHeight(90)
        self.setObjectName("TrackControl")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 상단: 이름 + 분리 버튼
        top_layout = QHBoxLayout()
        
        self.lbl_name = QLabel(self.track.name)
        self.lbl_name.setStyleSheet("font-weight: bold; color: #eee; background: transparent;")
        top_layout.addWidget(self.lbl_name)
        
        btn_sep = QPushButton("✂")
        btn_sep.setToolTip("Prompt Split")
        btn_sep.setFixedSize(25, 20)
        btn_sep.setStyleSheet("background-color: #555; color: white; border: none;")
        btn_sep.clicked.connect(self.request_separation)
        top_layout.addWidget(btn_sep)
        
        layout.addLayout(top_layout)
        layout.addStretch()

        # 하단: 오디오 컨트롤 (M/S/Vol)
        ctrl_layout = QHBoxLayout()
        
        self.btn_mute = QPushButton("M")
        self.btn_mute.setCheckable(True)
        self.btn_mute.setFixedSize(25, 25)
        self.btn_mute.setChecked(self.track.mute)
        self.btn_mute.toggled.connect(self.mute_toggled.emit)
        self.btn_mute.setStyleSheet("""
            QPushButton { background-color: #444; color: #aaa; border: 1px solid #555; }
            QPushButton:checked { background-color: #d32f2f; color: white; border: 1px solid #d32f2f; }
        """)
        
        self.btn_solo = QPushButton("S")
        self.btn_solo.setCheckable(True)
        self.btn_solo.setFixedSize(25, 25)
        self.btn_solo.setChecked(self.track.solo)
        self.btn_solo.toggled.connect(self.solo_toggled.emit)
        self.btn_solo.setStyleSheet("""
            QPushButton { background-color: #444; color: #aaa; border: 1px solid #555; }
            QPushButton:checked { background-color: #fbc02d; color: black; border: 1px solid #fbc02d; }
        """)
        
        self.slider_vol = QSlider(Qt.Orientation.Horizontal)
        self.slider_vol.setRange(0, 150)
        self.slider_vol.setValue(int(self.track.volume * 100))
        # 슬라이더 이동 시 시그널 발생
        self.slider_vol.valueChanged.connect(lambda v: self.volume_changed.emit(v/100.0))
        self.slider_vol.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #333; border-radius: 2px; }
            QSlider::handle:horizontal { background: #00aaff; width: 10px; margin: -3px 0; border-radius: 5px; }
            QSlider::sub-page:horizontal { background: #0088cc; border-radius: 2px; }
        """)

        ctrl_layout.addWidget(self.btn_mute)
        ctrl_layout.addWidget(self.btn_solo)
        ctrl_layout.addWidget(self.slider_vol)
        layout.addLayout(ctrl_layout)

    def request_separation(self):
        text, ok = QInputDialog.getText(self, "Prompt Separation", "추출할 소리 (예: Drums, Piano):")
        if ok and text:
            self.separationRequested.emit(self.track.id, text)

    def set_playhead_ratio(self, ratio):
        self.playhead_ratio = ratio
        self.update()

    def set_selected(self, selected: bool):
        self.setProperty("selected", selected)
        self.update() # 스타일 갱신

    def mousePressEvent(self, e):
        self.clicked.emit(self.track.id)
        super().mousePressEvent(e)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width(); h = self.height()

        # 배경색
        bg_color = QColor(40, 40, 40)
        if self.property("selected"): bg_color = QColor(60, 60, 75)
        painter.fillRect(self.rect(), bg_color)

        # 파형 그리기
        if self.track.data is not None and len(self.track.data) > 0:
            data = self.track.data
            step = max(1, len(data) // w)
            painter.setPen(QPen(QColor(120, 120, 120, 150), 1))
            
            if data.ndim > 1: samples = data[::step, 0]
            else: samples = data[::step]
            
            mid_y = h / 2
            draw_count = min(len(samples), w)
            for x in range(draw_count):
                val = samples[x]
                amp = abs(val) * (h * 0.4) 
                painter.drawLine(x, int(mid_y - amp), x, int(mid_y + amp))

        # 재생 헤드
        if self.playhead_ratio > 0:
            pos_x = int(w * self.playhead_ratio)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 30))
            painter.drawRect(0, 0, pos_x, h)
            painter.setPen(QPen(QColor(255, 50, 50), 1))
            painter.drawLine(pos_x, 0, pos_x, h)

        # 테두리
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(0, 0, w-1, h-1)


class InspectorPanel(QWidget):
    """
    우측 트랙 속성 패널
    [FIX] 시그널을 추가하여 변경 사항을 메인 윈도우로 전파
    [FIX] 'pan' 속성 대신 'angle_deg' 속성 사용
    """
    track_modified = pyqtSignal()   # 이름 변경 등 (타임라인 갱신용)
    volume_changed = pyqtSignal(float) # 볼륨 변경 (UI 동기화용)
    pan_changed = pyqtSignal(float)    # 팬 변경 (UI 동기화용)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.current_track = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. Track Info
        gb_info = QGroupBox("Track Info")
        f_info = QVBoxLayout()
        
        self.txt_name = QLineEdit()
        self.txt_name.editingFinished.connect(self.update_name)
        f_info.addWidget(QLabel("Name:"))
        f_info.addWidget(self.txt_name)
        gb_info.setLayout(f_info)
        layout.addWidget(gb_info)
        
        # 2. Audio Control
        gb_audio = QGroupBox("Audio Control")
        f_audio = QVBoxLayout()
        
        f_audio.addWidget(QLabel("Volume"))
        self.slider_vol = QSlider(Qt.Orientation.Horizontal)
        self.slider_vol.setRange(0, 150)
        self.slider_vol.valueChanged.connect(self.update_volume)
        f_audio.addWidget(self.slider_vol)
        
        f_audio.addWidget(QLabel("Pan (L/R)"))
        self.dial_pan = QDial()
        self.dial_pan.setRange(0, 200)
        self.dial_pan.setValue(100)
        self.dial_pan.setNotchesVisible(True)
        self.dial_pan.setFixedHeight(80)
        self.dial_pan.valueChanged.connect(self.update_pan)
        f_audio.addWidget(self.dial_pan)
        
        gb_audio.setLayout(f_audio)
        layout.addWidget(gb_audio)
        
        layout.addStretch()

    def set_track(self, track):
        self.current_track = track
        if not track:
            self.setEnabled(False)
            self.txt_name.clear()
            return
            
        self.setEnabled(True)
        self.blockSignals(True)
        self.txt_name.setText(track.name)
        self.slider_vol.setValue(int(track.volume * 100))
        
        # [수정] pan 대신 angle_deg 사용
        # angle_deg: -90(Left) ~ 0(Center) ~ 90(Right) 가정 (또는 270~0~90)
        # UI Dial: 0(Left) ~ 100(Center) ~ 200(Right)
        
        current_angle = track.angle_deg
        # 360도 체계일 경우 270도(-90도) 보정
        if current_angle > 180: 
            current_angle -= 360
            
        # -90 ~ 90 범위를 -1.0 ~ 1.0 비율로 변환
        pan_ratio = max(-1.0, min(1.0, current_angle / 90.0))
        
        # 비율을 Dial 값(0~200)으로 변환
        dial_val = int((pan_ratio + 1.0) * 100)
        self.dial_pan.setValue(dial_val)
        
        self.blockSignals(False)

    def update_name(self):
        if self.current_track:
            self.current_track.name = self.txt_name.text()
            self.track_modified.emit()

    def update_volume(self, val):
        if self.current_track:
            vol = val / 100.0
            self.current_track.volume = vol
            self.volume_changed.emit(vol) 

    def update_pan(self, val):
        if self.current_track:
            # 0~200 -> -1.0 ~ 1.0
            pan_ratio = (val - 100) / 100.0
            
            # 비율 -> 각도 (-90 ~ 90)
            angle = pan_ratio * 90.0
            self.current_track.angle_deg = angle
            
            self.pan_changed.emit(pan_ratio)