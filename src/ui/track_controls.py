import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, 
    QSlider, QDial, QMenu, QInputDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen

# [1] 웨이브폼 위젯
class WaveformWidget(QWidget):
    def __init__(self, vis_min, vis_max, parent=None):
        super().__init__(parent)
        self.vis_min = vis_min; self.vis_max = vis_max
        self.playhead_ratio = 0.0
        self.setStyleSheet("background-color: #222;")

    def set_playhead_ratio(self, r):
        self.playhead_ratio = max(0.0, min(1.0, float(r)))
        self.update()
            
    def paintEvent(self, event):
        if self.vis_min is None: return
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        p.setPen(QPen(QColor(0, 200, 255), 1))
        w, h = self.width(), self.height(); mid = h / 2.0
        n = len(self.vis_min); step = max(1, n // max(1, w))
        for x in range(w):
            idx = x * step
            if idx >= n: break
            p.drawLine(x, int(mid - (self.vis_max[idx]*mid*0.9)), x, int(mid - (self.vis_min[idx]*mid*0.9)))
        xh = int(self.playhead_ratio * max(1, w-1))
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawLine(xh, 0, xh, h)
        p.end()

# [2] 트랙 컨트롤 (기존 TrackStrip + 우클릭 메뉴 통합)
class TrackControls(QFrame):
    clicked = pyqtSignal(str)
    separationRequested = pyqtSignal(str, str) # AI 분리 신호

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.setFixedHeight(120)
        self.setStyleSheet("QFrame { background-color: #333; border: 1px solid #555; border-radius: 5px; }")
        
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        layout = QHBoxLayout(self); layout.setContentsMargins(5,5,5,5)
        
        info_layout = QVBoxLayout()
        self.lbl = QLabel(track.name)
        self.lbl.setStyleSheet("color:white; font-weight:bold; border:none; background:transparent;")
        info_layout.addWidget(self.lbl)
        
        btn_lo = QHBoxLayout()
        self.btn_mute = QPushButton("M"); self.btn_mute.setCheckable(True); self.btn_mute.setFixedSize(30,30)
        self.btn_mute.setStyleSheet("QPushButton{background:#444; color:white} QPushButton:checked{background:#f44;}")
        self.btn_mute.setChecked(self.track.mute)
        self.btn_mute.toggled.connect(lambda c: setattr(self.track, 'mute', c))
        
        self.btn_solo = QPushButton("S"); self.btn_solo.setCheckable(True); self.btn_solo.setFixedSize(30,30)
        self.btn_solo.setStyleSheet("QPushButton{background:#444; color:white} QPushButton:checked{background:#fd4; color:black;}")
        self.btn_solo.setChecked(self.track.solo)
        self.btn_solo.toggled.connect(lambda c: setattr(self.track, 'solo', c))
        
        btn_lo.addWidget(self.btn_mute); btn_lo.addWidget(self.btn_solo)
        info_layout.addLayout(btn_lo)
        layout.addLayout(info_layout)
        
        self.wave = WaveformWidget(track.vis_min, track.vis_max)
        layout.addWidget(self.wave, 1)
    
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.clicked.emit(self.track.id)
        super().mousePressEvent(e)

    def set_selected(self, s):
        self.setStyleSheet(f"QFrame {{ background-color: {'#444' if s else '#333'}; border: 2px solid {'#00aaff' if s else '#555'}; border-radius: 5px; }}")

    def set_playhead_ratio(self, r): self.wave.set_playhead_ratio(r)

    def show_context_menu(self, pos):
        menu = QMenu(self)
        act_rename = menu.addAction("이름 변경")
        menu.addSeparator()
        act_sep = menu.addAction("AI: 텍스트 프롬프트로 소리 추출 (Beta)")
        
        action = menu.exec(self.mapToGlobal(pos))
        if action == act_rename:
            new_name, ok = QInputDialog.getText(self, "이름 변경", "새 이름:", text=self.track.name)
            if ok: self.track.name = new_name; self.lbl.setText(new_name)
        elif action == act_sep:
            prompt, ok = QInputDialog.getText(self, "AI 소리 추출", "추출할 소리 (영어):\nExample: Guitar, Applause, Dog barking", text="")
            if ok and prompt: self.separationRequested.emit(self.track.id, prompt)

# [3] 인스펙터 패널 (오류 해결의 핵심)
class InspectorPanel(QFrame):
    track_modified = pyqtSignal()
    
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.current_track = None
        self.setMinimumWidth(300)
        self.setStyleSheet("QFrame { background-color: #2b2b2b; color: white; border: none; } QLabel { color: white; }")
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("🎚 Track Mixer"))
        self.lbl_name = QLabel("No Selection")
        self.lbl_name.setStyleSheet("font-size: 16px; font-weight: bold; color: #00aaff;")
        layout.addWidget(self.lbl_name)
        
        h = QHBoxLayout()
        self.sld_vol = QSlider(Qt.Orientation.Horizontal); self.sld_vol.setRange(0, 150)
        self.sld_vol.valueChanged.connect(lambda v: self._set_param('volume', v/100.0))
        h.addWidget(QLabel("Vol")); h.addWidget(self.sld_vol)
        layout.addLayout(h)
        
        self.dial_pan = QDial(); self.dial_pan.setRange(0, 360); self.dial_pan.setWrapping(True); self.dial_pan.setNotchesVisible(True); self.dial_pan.setFixedSize(80, 80)
        self.dial_pan.valueChanged.connect(lambda v: self._set_param('angle_deg', v))
        layout.addWidget(QLabel("Pan")); layout.addWidget(self.dial_pan)
        self.lbl_pan = QLabel("0°"); layout.addWidget(self.lbl_pan)
        
        layout.addStretch()

    def _set_param(self, p, v):
        if self.current_track:
            setattr(self.current_track, p, v)
            if p == 'angle_deg': self.lbl_pan.setText(f"{int(v)}°")
            self.track_modified.emit()

    def set_track(self, t):
        self.current_track = t
        if t:
            self.lbl_name.setText(t.name)
            self.sld_vol.setValue(int(t.volume*100))
            self.dial_pan.setValue(int(t.angle_deg))
            self.lbl_pan.setText(f"{int(t.angle_deg)}°")
        else:
            self.lbl_name.setText("None")