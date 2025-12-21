import sys
import os
import math
import time
from typing import Optional
import numpy as np
import librosa 

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLabel, QScrollArea, QFrame,
    QProgressBar, QMessageBox, QCheckBox, QTabWidget, QInputDialog,
    QGridLayout, QDoubleSpinBox, QSizePolicy, QDialog, QFormLayout, 
    QComboBox, QDialogButtonBox, QSpinBox, QStackedLayout, QColorDialog,
    QDockWidget, QGroupBox, QFontComboBox
)
from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal, QRect, QPointF, QRectF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QFont, QAction, QBrush, QFontMetrics, 
    QPixmap, QImage, QPolygonF, QLinearGradient, QFontDatabase
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink, QVideoFrame

try:
    from .models import ProjectContext, Track, SourceType
    from .audio_engine import AudioEngine
    from .ai_service import AISeparationWorker, ExternalFileLoader
    from .media_util import is_video_file
    from .logging_util import init_logging
    from .cache_bootstrap import ensure_writable_cache, apply_env_for_cache
    from .ffmpeg_util import extract_audio_to_wav, find_ffmpeg, merge_audio_to_video
    import tempfile
except ImportError:
    pass

def show_alert(parent, title: str, text: str, is_error: bool = False):
    mb = QMessageBox(parent)
    icon = QMessageBox.Icon.Critical if is_error else QMessageBox.Icon.Information
    mb.setIcon(icon)
    mb.setWindowTitle(title)
    mb.setText(text)
    mb.setStyleSheet("""
        QMessageBox { background-color: #2b2b2b; }
        QMessageBox QLabel { color: #eee; font-size: 13px; font-weight: bold; }
        QPushButton { color: #eee; background-color: #444; border: 1px solid #666; padding: 5px 20px; border-radius: 4px; }
        QPushButton:hover { background-color: #555; }
    """)
    mb.exec()

# --- Visualizer Settings Dialog ---
class VisualizerSettingsDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("🎨 Visualizer Settings")
        self.resize(500, 700)
        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: white; }
            QLabel, QCheckBox { color: white; }
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 10px; padding-top: 10px; color: #00ccff; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
            QPushButton { background: #444; color: white; padding: 5px; border-radius: 3px; border: 1px solid #666; }
            QComboBox, QSpinBox, QDoubleSpinBox, QFontComboBox { background: #444; color: white; padding: 4px; border: 1px solid #555; }
        """)
        
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab_spectrum = QWidget()
        self._init_spectrum_tab()
        self.tabs.addTab(self.tab_spectrum, "📊 Spectrum")

        self.tab_text = QWidget()
        self._init_text_tab()
        self.tabs.addTab(self.tab_text, "📝 Text Overlay")

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        main_layout.addWidget(self.buttons)

        if current_settings:
            self._load_settings(current_settings)

    def _init_spectrum_tab(self):
        layout = QFormLayout(self.tab_spectrum)
        
        self.sld_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sld_opacity.setRange(0, 100)
        
        self.combo_shape = QComboBox()
        self.combo_shape.addItems(["Linear (선형)", "Circular (원형)", "Polygon (다각형)"])

        self.combo_style = QComboBox()
        self.combo_style.addItems(["Bar (막대)", "Line (선)"])
        
        self.spin_thick = QSpinBox()
        self.spin_thick.setRange(1, 100)

        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.1, 20.0)
        self.spin_scale.setSingleStep(0.1)
        self.spin_scale.setSuffix(" x")

        # [Height Scale]
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.1, 100.0) 
        self.spin_height.setSingleStep(0.5)
        self.spin_height.setSuffix(" x")

        self.btn_color_bot = QPushButton("🎨 Bottom Color")
        self.btn_color_top = QPushButton("🎨 Top Color")
        self.btn_color_bot.clicked.connect(lambda: self.pick_color('bot'))
        self.btn_color_top.clicked.connect(lambda: self.pick_color('top'))
        self.color_bot = QColor(0, 0, 255)
        self.color_top = QColor(0, 255, 255)

        self.chk_mask = QCheckBox("Mask Mode (검은 배경 + 스펙트럼 투과)")
        self.btn_overlay_color = QPushButton("🎨 Overlay Color (Mask BG)")
        self.btn_overlay_color.clicked.connect(lambda: self.pick_color('overlay'))
        self.color_overlay = QColor(0, 0, 0)

        layout.addRow("🖼 BG Opacity:", self.sld_opacity)
        layout.addRow("📐 Shape:", self.combo_shape)
        layout.addRow("🎨 Style:", self.combo_style)
        layout.addRow("✏ Thickness:", self.spin_thick)
        layout.addRow("🎛 Sensitivity:", self.spin_scale)
        layout.addRow("📏 Max Height:", self.spin_height) 
        layout.addRow("🌈 Grad Start:", self.btn_color_bot)
        layout.addRow("🌈 Grad End:", self.btn_color_top)
        layout.addRow(self.chk_mask)
        layout.addRow("⬛ Mask BG Color:", self.btn_overlay_color)

    def _init_text_tab(self):
        layout = QVBoxLayout(self.tab_text)
        
        # [FIXED] Use QHBoxLayout instead of addRow (which is for FormLayout)
        h_opacity = QHBoxLayout()
        h_opacity.addWidget(QLabel("Text Opacity:"))
        self.sld_text_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sld_text_opacity.setRange(0, 100)
        self.sld_text_opacity.setValue(100)
        h_opacity.addWidget(self.sld_text_opacity)
        layout.addLayout(h_opacity)

        self.chk_text_mask = QCheckBox("Mask Text (텍스트 투명화/배경 비침)")
        layout.addWidget(self.chk_text_mask)

        # Title Group
        grp_title = QGroupBox("Main Title")
        form_t = QFormLayout(grp_title)
        self.edit_title = QComboBox() 
        self.edit_title.setEditable(True)
        self.edit_title.setEditText("Now Playing")
        
        self.font_title = QFontComboBox()
        self.spin_title_size = QSpinBox()
        self.spin_title_size.setRange(8, 500)
        self.spin_title_size.setValue(60)
        
        self.spin_title_x = QSpinBox()
        self.spin_title_x.setRange(0, 3000)
        self.spin_title_x.setValue(50)
        self.spin_title_y = QSpinBox()
        self.spin_title_y.setRange(0, 2000)
        self.spin_title_y.setValue(100)

        self.chk_title_bold = QCheckBox("Bold")
        self.chk_title_italic = QCheckBox("Italic")
        self.btn_title_color = QPushButton("Color")
        self.color_title = QColor(255, 255, 255)
        self.btn_title_color.clicked.connect(lambda: self.pick_text_color('title'))
        
        form_t.addRow("Text:", self.edit_title)
        form_t.addRow("Font:", self.font_title)
        form_t.addRow("Size:", self.spin_title_size)
        form_t.addRow("Pos X:", self.spin_title_x)
        form_t.addRow("Pos Y:", self.spin_title_y)
        form_t.addRow("Style:", self.chk_title_bold)
        form_t.addRow("", self.chk_title_italic)
        form_t.addRow("Color:", self.btn_title_color)
        layout.addWidget(grp_title)

        # Subtitle Group
        grp_sub = QGroupBox("Subtitle")
        form_s = QFormLayout(grp_sub)
        self.edit_sub = QComboBox()
        self.edit_sub.setEditable(True)
        self.edit_sub.setEditText("Artist Name")
        
        self.font_sub = QFontComboBox()
        self.spin_sub_size = QSpinBox()
        self.spin_sub_size.setRange(8, 500)
        self.spin_sub_size.setValue(30)

        self.spin_sub_x = QSpinBox()
        self.spin_sub_x.setRange(0, 3000)
        self.spin_sub_x.setValue(50)
        self.spin_sub_y = QSpinBox()
        self.spin_sub_y.setRange(0, 2000)
        self.spin_sub_y.setValue(160)

        self.chk_sub_bold = QCheckBox("Bold")
        self.chk_sub_italic = QCheckBox("Italic")
        self.btn_sub_color = QPushButton("Color")
        self.color_sub = QColor(200, 200, 200)
        self.btn_sub_color.clicked.connect(lambda: self.pick_text_color('sub'))

        form_s.addRow("Text:", self.edit_sub)
        form_s.addRow("Font:", self.font_sub)
        form_s.addRow("Size:", self.spin_sub_size)
        form_s.addRow("Pos X:", self.spin_sub_x)
        form_s.addRow("Pos Y:", self.spin_sub_y)
        form_s.addRow("Style:", self.chk_sub_bold)
        form_s.addRow("", self.chk_sub_italic)
        form_s.addRow("Color:", self.btn_sub_color)
        layout.addWidget(grp_sub)
        layout.addStretch()

    def _load_settings(self, s):
        self.sld_opacity.setValue(int(s.get('bg_opacity', 1.0) * 100))
        shape_map = {'linear': 0, 'circular': 1, 'polygon': 2}
        self.combo_shape.setCurrentIndex(shape_map.get(s.get('shape', 'linear'), 0))
        self.combo_style.setCurrentIndex(0 if s.get('draw_mode', 'bar') == 'bar' else 1)
        self.spin_thick.setValue(s.get('thickness', 5))
        self.spin_scale.setValue(s.get('scale', 2.0))
        self.spin_height.setValue(s.get('height_scale', 1.0))
        
        c_bot = s.get('color_bot')
        c_top = s.get('color_top')
        c_ov = s.get('color_overlay')
        if c_bot: self.color_bot = c_bot
        if c_top: self.color_top = c_top
        if c_ov: self.color_overlay = c_ov
        self.update_color_buttons()
        self.chk_mask.setChecked(s.get('mask_mode', False))

        # Text Settings
        self.sld_text_opacity.setValue(int(s.get('text_opacity', 1.0) * 100))
        self.chk_text_mask.setChecked(s.get('text_mask_mode', False))
        
        t_set = s.get('title', {})
        self.edit_title.setEditText(t_set.get('text', ''))
        self.font_title.setCurrentFont(QFont(t_set.get('font', 'Arial')))
        self.spin_title_size.setValue(t_set.get('size', 48))
        self.spin_title_x.setValue(t_set.get('x', 50))
        self.spin_title_y.setValue(t_set.get('y', 100))
        self.chk_title_bold.setChecked(t_set.get('bold', True))
        self.chk_title_italic.setChecked(t_set.get('italic', False))
        if t_set.get('color'): self.color_title = t_set.get('color')
        self.btn_title_color.setStyleSheet(f"background: {self.color_title.name()}; color: black;")

        s_set = s.get('subtitle', {})
        self.edit_sub.setEditText(s_set.get('text', ''))
        self.font_sub.setCurrentFont(QFont(s_set.get('font', 'Arial')))
        self.spin_sub_size.setValue(s_set.get('size', 24))
        self.spin_sub_x.setValue(s_set.get('x', 50))
        self.spin_sub_y.setValue(s_set.get('y', 160))
        self.chk_sub_bold.setChecked(s_set.get('bold', False))
        self.chk_sub_italic.setChecked(s_set.get('italic', False))
        if s_set.get('color'): self.color_sub = s_set.get('color')
        self.btn_sub_color.setStyleSheet(f"background: {self.color_sub.name()}; color: black;")

    def pick_color(self, target):
        if target == 'bot': init = self.color_bot
        elif target == 'top': init = self.color_top
        else: init = self.color_overlay

        c = QColorDialog.getColor(init, self, f"Pick Color")
        if c.isValid():
            if target == 'bot': self.color_bot = c
            elif target == 'top': self.color_top = c
            else: self.color_overlay = c
            self.update_color_buttons()

    def update_color_buttons(self):
        self.btn_color_bot.setStyleSheet(f"background-color: {self.color_bot.name()}; color: {'black' if self.color_bot.lightness() > 128 else 'white'};")
        self.btn_color_top.setStyleSheet(f"background-color: {self.color_top.name()}; color: {'black' if self.color_top.lightness() > 128 else 'white'};")
        self.btn_overlay_color.setStyleSheet(f"background-color: {self.color_overlay.name()}; color: {'black' if self.color_overlay.lightness() > 128 else 'white'};")

    def pick_text_color(self, target):
        init = self.color_title if target == 'title' else self.color_sub
        c = QColorDialog.getColor(init, self, "Pick Text Color")
        if c.isValid():
            if target == 'title': 
                self.color_title = c
                self.btn_title_color.setStyleSheet(f"background: {c.name()}; color: black;")
            else: 
                self.color_sub = c
                self.btn_sub_color.setStyleSheet(f"background: {c.name()}; color: black;")

    def get_settings(self):
        shape_idx = self.combo_shape.currentIndex()
        shapes = ['linear', 'circular', 'polygon']
        return {
            'bg_opacity': self.sld_opacity.value() / 100.0,
            'shape': shapes[shape_idx],
            'draw_mode': 'bar' if self.combo_style.currentIndex() == 0 else 'line',
            'thickness': self.spin_thick.value(),
            'scale': self.spin_scale.value(),
            'height_scale': self.spin_height.value(),
            'color_bot': self.color_bot,
            'color_top': self.color_top,
            'color_overlay': self.color_overlay,
            'mask_mode': self.chk_mask.isChecked(),
            'text_opacity': self.sld_text_opacity.value() / 100.0,
            'text_mask_mode': self.chk_text_mask.isChecked(),
            'title': {
                'text': self.edit_title.currentText(),
                'font': self.font_title.currentFont().family(),
                'size': self.spin_title_size.value(),
                'x': self.spin_title_x.value(),
                'y': self.spin_title_y.value(),
                'bold': self.chk_title_bold.isChecked(),
                'italic': self.chk_title_italic.isChecked(),
                'color': self.color_title
            },
            'subtitle': {
                'text': self.edit_sub.currentText(),
                'font': self.font_sub.currentFont().family(),
                'size': self.spin_sub_size.value(),
                'x': self.spin_sub_x.value(),
                'y': self.spin_sub_y.value(),
                'bold': self.chk_sub_bold.isChecked(),
                'italic': self.chk_sub_italic.isChecked(),
                'color': self.color_sub
            }
        }

# --- Visualizer Screen (Unified) ---
class VisualizerScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setMinimumHeight(200) 
        
        self.video_sink = QVideoSink()
        self.video_sink.videoFrameChanged.connect(self._on_video_frame)
        self.current_video_frame: Optional[QImage] = None
        
        self.audio_data = None
        self.bar_count = 60
        self.decay = np.zeros(self.bar_count)
        
        self.bg_pixmap: Optional[QPixmap] = None
        self.use_bg_image = False
        
        # Visual Settings
        self.show_spectrum = False
        self.bg_opacity = 1.0
        self.shape = 'linear'
        self.draw_mode = 'bar'
        self.thickness = 5
        self.scale = 2.0
        self.height_scale = 1.0
        self.color_bot = QColor(0, 0, 255)
        self.color_top = QColor(0, 255, 255)
        self.color_overlay = QColor(0, 0, 0)
        self.mask_mode = False 
        
        # Text Settings
        self.text_opacity = 1.0
        self.text_mask_mode = False
        self.title_cfg = {'text': '', 'size': 60, 'x': 50, 'y': 100, 'color': QColor('white')}
        self.sub_cfg = {'text': '', 'size': 30, 'x': 50, 'y': 160, 'color': QColor('lightgray')}

    def _on_video_frame(self, frame: QVideoFrame):
        if frame.isValid():
            self.current_video_frame = frame.toImage()
            self.update()

    def set_settings(self, s):
        self.bg_opacity = s.get('bg_opacity', 1.0)
        self.shape = s.get('shape', 'linear')
        self.draw_mode = s.get('draw_mode', 'bar')
        self.thickness = s.get('thickness', 5)
        self.scale = s.get('scale', 2.0)
        self.height_scale = s.get('height_scale', 1.0)
        self.color_bot = s.get('color_bot', QColor(0, 0, 255))
        self.color_top = s.get('color_top', QColor(0, 255, 255))
        self.color_overlay = s.get('color_overlay', QColor(0, 0, 0))
        self.mask_mode = s.get('mask_mode', False)
        
        self.text_opacity = s.get('text_opacity', 1.0)
        self.text_mask_mode = s.get('text_mask_mode', False)
        self.title_cfg = s.get('title', {})
        self.sub_cfg = s.get('subtitle', {})
        self.update()

    def get_settings(self):
        return {
            'bg_opacity': self.bg_opacity,
            'shape': self.shape,
            'draw_mode': self.draw_mode,
            'thickness': self.thickness,
            'scale': self.scale,
            'height_scale': self.height_scale,
            'color_bot': self.color_bot,
            'color_top': self.color_top,
            'color_overlay': self.color_overlay,
            'mask_mode': self.mask_mode,
            'text_opacity': self.text_opacity,
            'text_mask_mode': self.text_mask_mode,
            'title': self.title_cfg,
            'subtitle': self.sub_cfg
        }

    def set_audio_data(self, data: np.ndarray):
        if data is None: self.audio_data = None
        else: self.audio_data = np.mean(data, axis=1)
        self.update()

    def set_background_image(self, path: str):
        if not path:
            self.bg_pixmap = None
            self.use_bg_image = False
        else:
            self.bg_pixmap = QPixmap(path)
            self.use_bg_image = True
            self.current_video_frame = None
        self.update()

    def clear_background(self):
        self.use_bg_image = False
        self.current_video_frame = None
        self.update()

    def toggle_spectrum(self, show: bool):
        self.show_spectrum = show
        self.update()

    def _process_fft(self):
        if self.audio_data is None or len(self.audio_data) < 100: 
            return np.zeros(self.bar_count)
        
        windowed = self.audio_data * np.hanning(len(self.audio_data))
        fft = np.fft.rfft(windowed)
        mag = np.abs(fft)
        mag = mag * 100.0 

        if len(mag) > self.bar_count:
            step = len(mag) // self.bar_count
            binned = np.array([np.mean(mag[i*step:(i+1)*step]) for i in range(self.bar_count)])
        else:
            binned = np.zeros(self.bar_count)

        binned = np.log10(binned + 1.0) * 300.0 * self.scale
        self.decay = np.maximum(self.decay * 0.80, binned)
        return self.decay

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        # 1. Background
        p.fillRect(0, 0, w, h, QColor(0, 0, 0))

        if self.use_bg_image and self.bg_pixmap:
            p.setOpacity(self.bg_opacity)
            scaled = self.bg_pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            sx = (w - scaled.width()) // 2
            sy = (h - scaled.height()) // 2
            p.drawPixmap(sx, sy, scaled)
            p.setOpacity(1.0)
        elif self.current_video_frame:
            p.setOpacity(self.bg_opacity)
            scaled = self.current_video_frame.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            sx = (w - scaled.width()) // 2
            sy = (h - scaled.height()) // 2
            p.drawImage(sx, sy, scaled)
            p.setOpacity(1.0)

        # 2. Composition (Spectrum & Text)
        need_masking = self.mask_mode or self.text_mask_mode
        
        if need_masking:
            buffer = QPixmap(w, h)
            buffer.fill(Qt.GlobalColor.transparent)
            pb = QPainter(buffer)
            pb.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            if self.mask_mode:
                pb.fillRect(0, 0, w, h, self.color_overlay)
            
            # Draw Spectrum
            if self.show_spectrum:
                decay_data = self._process_fft()
                draw_w = w * 0.6
                draw_h = h * 0.5
                draw_x = (w - draw_w) / 2
                draw_y = (h - draw_h) / 2
                draw_rect = QRectF(draw_x, draw_y, draw_w, draw_h)
                
                if self.mask_mode:
                    pb.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
                    self._draw_shape(pb, draw_rect, decay_data, is_mask=True)
                    pb.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                else:
                    self._draw_shape(pb, draw_rect, decay_data, is_mask=False)

            # Draw Text
            if self.text_mask_mode:
                pb.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
                self._draw_text_internal(pb, w, h, is_mask=True)
                pb.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            else:
                self._draw_text_internal(pb, w, h, is_mask=False)

            pb.end()
            p.drawPixmap(0, 0, buffer)

        else:
            if self.show_spectrum:
                decay_data = self._process_fft()
                draw_w = w * 0.6
                draw_h = h * 0.5
                draw_x = (w - draw_w) / 2
                draw_y = (h - draw_h) / 2
                draw_rect = QRectF(draw_x, draw_y, draw_w, draw_h)
                self._draw_shape(p, draw_rect, decay_data, is_mask=False)
            
            self._draw_text_internal(p, w, h, is_mask=False)

        p.end()

    def _draw_shape(self, p: QPainter, rect: QRectF, data, is_mask=False):
        if is_mask:
            brush = QBrush(QColor(255, 255, 255, 255))
            pen_color = QColor(255, 255, 255, 255)
        else:
            gradient = QLinearGradient(rect.bottomLeft(), rect.topLeft())
            gradient.setColorAt(0.0, self.color_bot)
            gradient.setColorAt(1.0, self.color_top)
            brush = QBrush(gradient)
            pen_color = self.color_top

        if self.draw_mode == 'bar':
            p.setBrush(brush)
            # [FIXED] Force Visible Pen for Circular Bar
            if self.shape != 'linear':
                pen = QPen(pen_color)
                pen.setWidth(self.thickness)
                p.setPen(pen)
            else:
                p.setPen(Qt.PenStyle.NoPen)
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(pen_color)
            pen.setWidth(self.thickness)
            p.setPen(pen)

        scaled_data = data * self.height_scale

        if self.shape == 'linear':
            self._draw_linear(p, rect, scaled_data)
        elif self.shape == 'circular':
            self._draw_circular(p, rect, scaled_data, is_polygon=False)
        elif self.shape == 'polygon':
            self._draw_circular(p, rect, scaled_data, is_polygon=True)

    def _draw_linear(self, p: QPainter, rect: QRectF, data):
        count = len(data)
        total_w = rect.width() / count
        h = rect.height()
        if self.draw_mode == 'bar':
            gap = max(0, int(total_w * (1 - (self.thickness / 50.0))))
            bar_w = max(1, total_w - gap)
            for i in range(count):
                bh = min(h * 5.0, data[i]) 
                x = rect.x() + i * total_w + (gap/2)
                y = rect.y() + h - bh
                p.drawRect(QRectF(x, y, bar_w, bh))
        else:
            points = []
            for i in range(count):
                bh = min(h * 5.0, data[i])
                x = rect.x() + i * total_w + (total_w/2)
                y = rect.y() + h - bh
                points.append(QPointF(x, y))
            if points: p.drawPolyline(points)

    def _draw_circular(self, p: QPainter, rect: QRectF, data, is_polygon=False):
        cx, cy = rect.center().x(), rect.center().y()
        max_r = min(rect.width(), rect.height()) / 2
        base_r = max_r * 0.3
        count = len(data)
        if count == 0: return
        angle_step = 2 * math.pi / count
        points = []
        
        for i in range(count):
            mag = data[i] 
            angle = -math.pi / 2 + (i * angle_step)
            r_out = base_r + mag
            x_start = cx + math.cos(angle) * base_r
            y_start = cy + math.sin(angle) * base_r
            x_end = cx + math.cos(angle) * r_out
            y_end = cy + math.sin(angle) * r_out
            
            if self.draw_mode == 'bar' and not is_polygon:
                bar_pen = p.pen()
                bar_pen.setWidth(self.thickness)
                if p.brush().style() != Qt.BrushStyle.NoBrush:
                     grad_pos = min(1.0, mag / max_r)
                     lerped_color = QColor(
                         int(self.color_bot.red() + (self.color_top.red() - self.color_bot.red()) * grad_pos),
                         int(self.color_bot.green() + (self.color_top.green() - self.color_bot.green()) * grad_pos),
                         int(self.color_bot.blue() + (self.color_top.blue() - self.color_bot.blue()) * grad_pos)
                     )
                     bar_pen.setColor(lerped_color)
                else:
                    bar_pen.setColor(self.color_top)
                
                p.setPen(bar_pen)
                p.drawLine(QPointF(x_start, y_start), QPointF(x_end, y_end))
            else:
                points.append(QPointF(x_end, y_end))
        
        if self.draw_mode == 'line' or is_polygon:
            if len(points) > 0: points.append(points[0]) 
            if is_polygon and self.draw_mode == 'bar':
                p.drawPolygon(QPolygonF(points))
            else:
                p.setBrush(Qt.BrushStyle.NoBrush)
                line_pen = p.pen()
                line_pen.setStyle(Qt.PenStyle.SolidLine)
                line_pen.setWidth(self.thickness)
                p.setPen(line_pen)
                p.drawPolyline(points)

    def _draw_text_internal(self, p: QPainter, w, h, is_mask=False):
        alpha = int(255 * self.text_opacity)
        
        # Title
        t_text = self.title_cfg.get('text', '')
        if t_text:
            font = QFont(self.title_cfg.get('font', 'Arial'), self.title_cfg.get('size', 48))
            font.setBold(self.title_cfg.get('bold', False))
            font.setItalic(self.title_cfg.get('italic', False))
            p.setFont(font)
            
            if is_mask:
                p.setPen(QColor(255, 255, 255, 255)) 
            else:
                c = self.title_cfg.get('color', QColor('white'))
                c.setAlpha(alpha)
                p.setPen(c)
            
            x = self.title_cfg.get('x', 50)
            y = self.title_cfg.get('y', 100)
            p.drawText(x, y, t_text)

        # Subtitle
        s_text = self.sub_cfg.get('text', '')
        if s_text:
            font = QFont(self.sub_cfg.get('font', 'Arial'), self.sub_cfg.get('size', 24))
            font.setBold(self.sub_cfg.get('bold', False))
            font.setItalic(self.sub_cfg.get('italic', False))
            p.setFont(font)
            
            if is_mask:
                p.setPen(QColor(255, 255, 255, 255))
            else:
                c = self.sub_cfg.get('color', QColor('lightgray'))
                c.setAlpha(alpha)
                p.setPen(c)
            
            x = self.sub_cfg.get('x', 50)
            y = self.sub_cfg.get('y', 160)
            p.drawText(x, y, s_text)

# --- Other Widgets ---
class CircularAngleDial(QWidget):
    angleChanged = pyqtSignal(float)
    def __init__(self, angle_deg: float = 0.0, parent=None):
        super().__init__(parent)
        self._angle = float(angle_deg) % 360.0
        self.setFixedSize(62, 62)
    def setAngle(self, deg: float):
        deg = float(deg) % 360.0
        if abs(deg - self._angle) > 0.01:
            self._angle = deg
            self.angleChanged.emit(self._angle)
            self.update()
    def _pos_to_angle(self, x: float, y: float) -> float:
        cx, cy = self.width() / 2.0, self.height() / 2.0
        return (math.degrees(math.atan2(x - cx, -(y - cy))) + 360.0) % 360.0
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.setAngle(self._pos_to_angle(e.position().x(), e.position().y()))
    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton: self.setAngle(self._pos_to_angle(e.position().x(), e.position().y()))
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        r = min(w, h) / 2.0 - 5
        p.setPen(QPen(QColor(100, 100, 100), 2))
        p.setBrush(QColor(40, 40, 40))
        p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        theta = math.radians(self._angle)
        x2 = cx + math.sin(theta) * (r - 8)
        y2 = cy - math.cos(theta) * (r - 8)
        p.setPen(QPen(QColor(0, 204, 255), 3))
        p.drawLine(int(cx), int(cy), int(x2), int(y2))
        p.setBrush(QColor(0, 204, 255))
        p.drawEllipse(int(x2 - 3), int(y2 - 3), 6, 6)
        p.setPen(QPen(QColor(220, 220, 220), 1))
        p.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{int(self._angle):03}°")
        p.end()

class WaveformWidget(QWidget):
    selectionChanged = pyqtSignal(float, float)
    def __init__(self, vis_min, vis_max, parent=None):
        super().__init__(parent)
        self.vis_min = vis_min
        self.vis_max = vis_max
        self.playhead_ratio = 0.0
        self.selection_start = 0.0
        self.selection_end = 0.0
        self.is_selecting = False
        self.drag_start_x = 0
        self.setFixedHeight(50)
        self.setStyleSheet("background-color: transparent;")
        self.setCursor(Qt.CursorShape.CrossCursor) 
    def set_playhead_ratio(self, r: float):
        self.playhead_ratio = max(0.0, min(1.0, float(r)))
        self.update()
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = True
            self.drag_start_x = e.position().x()
            ratio = self.drag_start_x / self.width()
            self.selection_start = ratio
            self.selection_end = ratio
            self.selectionChanged.emit(ratio, ratio)
            self.update()
    def mouseMoveEvent(self, e):
        if self.is_selecting:
            cur_x = e.position().x()
            ratio = max(0.0, min(1.0, cur_x / self.width()))
            start_ratio = self.drag_start_x / self.width()
            self.selection_start = min(start_ratio, ratio)
            self.selection_end = max(start_ratio, ratio)
            self.selectionChanged.emit(self.selection_start, self.selection_end)
            self.update()
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = False
            if abs(self.selection_end - self.selection_start) < 0.001:
                self.selection_start = 0.0
                self.selection_end = 0.0
                self.selectionChanged.emit(0.0, 0.0)
            self.update()
    def paintEvent(self, event):
        if self.vis_min is None: return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        p.setPen(QPen(QColor(0, 200, 255), 1))
        w, h, mid = self.width(), self.height(), self.height() / 2.0
        n = len(self.vis_min)
        step = max(1, n // max(1, w))
        for x in range(w):
            idx = x * step
            if idx >= n: break
            p.drawLine(x, int(mid - (self.vis_max[idx] * mid * 0.9)), x, int(mid - (self.vis_min[idx] * mid * 0.9)))
        if self.selection_end > self.selection_start:
            sx = int(self.selection_start * w)
            ex = int(self.selection_end * w)
            sel_rect = QRect(sx, 0, ex - sx, h)
            p.fillRect(sel_rect, QColor(255, 255, 0, 60))
            p.setPen(QPen(QColor(255, 255, 0), 1))
            p.drawRect(sel_rect)
        xh = int(self.playhead_ratio * max(1, w - 1))
        p.setPen(QPen(QColor(255, 255, 255, 200), 1))
        p.drawLine(xh, 0, xh, h)
        p.end()

class TrackStrip(QFrame):
    clicked = pyqtSignal(str)
    rename_requested = pyqtSignal(str, str)
    selection_changed = pyqtSignal(float, float) 
    def __init__(self, track: Track, parent=None):
        super().__init__(parent)
        self.track = track
        self._active = False
        self._selected = False
        self.setFixedHeight(100)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(5) 
        top = QHBoxLayout()
        top.setSpacing(10)
        self.lbl_name = QLabel(track.name)
        self.lbl_name.setFixedWidth(150) 
        self.lbl_name.setStyleSheet("QLabel { font-weight: bold; font-size: 13px; color: #ffffff; background-color: #333; padding: 4px; border-radius: 4px; border: 1px solid #444; }")
        self.lbl_name.setToolTip(track.name)
        self.lbl_name.mouseDoubleClickEvent = self.on_name_dbl_click
        self.lbl_name.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.wave = WaveformWidget(track.vis_min, track.vis_max)
        self.wave.selectionChanged.connect(self.selection_changed)
        top.addWidget(self.lbl_name)
        top.addWidget(self.wave, 1) 
        layout.addLayout(top)
        btm = QHBoxLayout()
        self.btn_mute = QPushButton("M")
        self.btn_mute.setCheckable(True)
        self.btn_mute.setFixedSize(40, 25)
        self.btn_mute.toggled.connect(lambda v: setattr(self.track, "mute", bool(v)))
        self.btn_mute.setStyleSheet("QPushButton { font-weight: bold; color: #ddd; } QPushButton:checked { background-color: #ff4444; color: white; }")
        self.btn_solo = QPushButton("S")
        self.btn_solo.setCheckable(True)
        self.btn_solo.setFixedSize(40, 25)
        self.btn_solo.toggled.connect(lambda v: setattr(self.track, "solo", bool(v)))
        self.btn_solo.setStyleSheet("QPushButton { font-weight: bold; color: #ddd; } QPushButton:checked { background-color: #ffdd44; color: black; }")
        btm.addStretch(1)
        btm.addWidget(self.btn_mute)
        btm.addWidget(self.btn_solo)
        layout.addLayout(btm)
        self._apply_style()
    def on_name_dbl_click(self, e):
        text, ok = QInputDialog.getText(self, "이름 변경", "새 트랙 이름:", text=self.track.name)
        if ok and text:
            self.track.name = text
            self.lbl_name.setText(text)
            self.lbl_name.setToolTip(text)
            self.rename_requested.emit(self.track.id, text)
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.clicked.emit(self.track.id)
    def set_playhead_ratio(self, r): self.wave.set_playhead_ratio(r)
    def set_active(self, a): 
        if self._active != a: 
            self._active = a
            self._apply_style()
    def set_selected(self, s):
        if self._selected != s:
            self._selected = s
            self._apply_style()
    def _apply_style(self):
        border = "#00aaff" if self._selected else "#555"
        bg = "#2a2a2a" if self._selected else "#222"
        self.setStyleSheet(f"QFrame {{ background-color: {bg}; border: {2 if self._selected else 1}px solid {border}; border-radius: 6px; }}")

class InspectorPanel(QFrame):
    track_modified = pyqtSignal()
    loop_request = pyqtSignal(float, float)
    process_started = pyqtSignal()
    process_finished = pyqtSignal()
    def __init__(self, ctx: ProjectContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.current_track: Optional[Track] = None
        self.setMinimumWidth(320)
        self.setStyleSheet("QFrame { background: #2b2b2b; color: #ffffff; border: none; } QLabel { color: #ffffff; font-size: 12px; } QSlider::groove:horizontal { background: #444; height: 6px; border-radius: 3px; } QSlider::handle:horizontal { background: #00ccff; width: 14px; margin: -5px 0; border-radius: 7px; } QTabWidget::pane { border: 1px solid #444; } QTabBar::tab { background: #333; color: #aaa; padding: 8px 12px; } QTabBar::tab:selected { background: #555; color: white; font-weight: bold; border-top: 2px solid #00ccff; }")
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tab_mix = QWidget()
        self.tab_fx = QWidget()
        self.tab_edit = QWidget()
        self.tabs.addTab(self.tab_mix, "Mixer")
        self.tabs.addTab(self.tab_fx, "EQ / FX")
        self.tabs.addTab(self.tab_edit, "Edit")
        main_layout.addWidget(self.tabs)
        self._init_mixer_tab()
        self._init_fx_tab()
        self._init_edit_tab()
        self.refresh_from_ctx()
    def update_selection(self, start_ratio, end_ratio):
        if not self.current_track: return
        duration = self.current_track.duration
        s_sec = start_ratio * duration
        e_sec = end_ratio * duration
        self.spin_cut_start.setValue(s_sec)
        self.spin_cut_end.setValue(e_sec)
    def _init_mixer_tab(self):
        layout = QVBoxLayout(self.tab_mix)
        grp = QFrame()
        grp.setStyleSheet("background: #383838; border-radius: 6px; padding: 8px; margin-bottom: 10px;")
        v = QVBoxLayout(grp)
        v.addWidget(QLabel("🎚 Master Control"))
        h = QHBoxLayout()
        self.sld_master_vol = QSlider(Qt.Orientation.Horizontal)
        self.sld_master_vol.setRange(0, 150)
        self.sld_master_vol.setValue(100)
        self.sld_master_vol.valueChanged.connect(lambda v: setattr(self.ctx, "master_volume", v/100.0))
        h.addWidget(QLabel("Vol"))
        h.addWidget(self.sld_master_vol)
        v.addLayout(h)
        h2 = QHBoxLayout()
        self.dial_master = CircularAngleDial()
        self.dial_master.angleChanged.connect(lambda d: setattr(self.ctx, "master_angle_deg", d))
        h2.addWidget(QLabel("Angle"))
        h2.addWidget(self.dial_master)
        v.addLayout(h2)
        layout.addWidget(grp)
        self.grp_track = QFrame()
        self.grp_track.setStyleSheet("background: #383838; border-radius: 6px; padding: 8px;")
        t = QVBoxLayout(self.grp_track)
        self.lbl_track_name = QLabel("No Track Selected")
        self.lbl_track_name.setStyleSheet("color: #00ccff; font-weight: bold; font-size: 14px;")
        t.addWidget(self.lbl_track_name)
        h3 = QHBoxLayout()
        self.sld_track_vol = QSlider(Qt.Orientation.Horizontal)
        self.sld_track_vol.setRange(0, 150)
        self.sld_track_vol.valueChanged.connect(lambda v: self.current_track and setattr(self.current_track, "volume", v/100.0))
        h3.addWidget(QLabel("Vol"))
        h3.addWidget(self.sld_track_vol)
        t.addLayout(h3)
        h4 = QHBoxLayout()
        self.dial_track = CircularAngleDial()
        self.dial_track.angleChanged.connect(lambda d: self.current_track and setattr(self.current_track, "angle_deg", d))
        h4.addWidget(QLabel("Angle"))
        h4.addWidget(self.dial_track)
        t.addLayout(h4)
        layout.addWidget(self.grp_track)
        layout.addStretch()
    def _init_fx_tab(self):
        layout = QVBoxLayout(self.tab_fx)
        grp_eq = QFrame()
        grp_eq.setStyleSheet("background: #383838; border-radius: 6px; padding: 5px;")
        v = QVBoxLayout(grp_eq)
        v.addWidget(QLabel("🎛 3-Band EQ"))
        grid = QGridLayout()
        self.sld_eq_low = QSlider(Qt.Orientation.Vertical)
        self.sld_eq_mid = QSlider(Qt.Orientation.Vertical)
        self.sld_eq_high = QSlider(Qt.Orientation.Vertical)
        for s in [self.sld_eq_low, self.sld_eq_mid, self.sld_eq_high]:
            s.setRange(0, 200)
            s.setValue(100)
            s.setFixedHeight(120) 
        self.sld_eq_low.valueChanged.connect(lambda v: self.current_track and setattr(self.current_track, "eq_low", v/100.0))
        self.sld_eq_mid.valueChanged.connect(lambda v: self.current_track and setattr(self.current_track, "eq_mid", v/100.0))
        self.sld_eq_high.valueChanged.connect(lambda v: self.current_track and setattr(self.current_track, "eq_high", v/100.0))
        grid.addWidget(self.sld_eq_low, 0, 0, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.sld_eq_mid, 0, 1, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.sld_eq_high, 0, 2, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QLabel("Low"), 1, 0, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QLabel("Mid"), 1, 1, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QLabel("High"), 1, 2, Qt.AlignmentFlag.AlignCenter)
        v.addLayout(grid)
        layout.addWidget(grp_eq)
        grp_pitch = QFrame()
        grp_pitch.setStyleSheet("background: #383838; border-radius: 6px; padding: 5px; margin-top: 10px;")
        pv = QVBoxLayout(grp_pitch)
        pv.addWidget(QLabel("🎤 Voice Pitch"))
        ph = QHBoxLayout()
        self.spin_pitch = QDoubleSpinBox()
        self.spin_pitch.setRange(-12.0, 12.0)
        self.spin_pitch.setValue(0.0)
        self.spin_pitch.setSuffix(" semi")
        self.spin_pitch.setMinimumHeight(35)
        self.spin_pitch.setStyleSheet("QDoubleSpinBox { font-size: 14px; background: #555; color: white; border: 1px solid #777; } QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 25px; }")
        self.btn_apply_pitch = QPushButton("Apply")
        self.btn_apply_pitch.setMinimumHeight(35)
        self.btn_apply_pitch.clicked.connect(self._apply_pitch_shift)
        ph.addWidget(self.spin_pitch)
        ph.addWidget(self.btn_apply_pitch)
        pv.addLayout(ph)
        layout.addWidget(grp_pitch)
        layout.addStretch()
    def _init_edit_tab(self):
        layout = QVBoxLayout(self.tab_edit)
        lbl_info = QLabel("💡 Tip: 파형 위에서 마우스를 드래그하여\n구간을 선택하세요.")
        lbl_info.setStyleSheet("color: #aaa; font-style: italic; margin-bottom: 10px;")
        layout.addWidget(lbl_info)
        grp_sel = QFrame()
        grp_sel.setStyleSheet("background: #383838; border-radius: 6px; padding: 10px;")
        v = QVBoxLayout(grp_sel)
        v.addWidget(QLabel("✂️ Selection Action"))
        g = QGridLayout()
        self.spin_cut_start = QDoubleSpinBox()
        self.spin_cut_end = QDoubleSpinBox()
        self.spin_cut_start.setRange(0, 9999)
        self.spin_cut_end.setRange(0, 9999)
        self.spin_cut_start.setSuffix(" s")
        self.spin_cut_end.setSuffix(" s")
        for spin in [self.spin_cut_start, self.spin_cut_end]:
            spin.setMinimumHeight(35)
            spin.setStyleSheet("QDoubleSpinBox { font-size: 14px; background: #555; color: white; border: 1px solid #777; } QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 25px; }")
        g.addWidget(QLabel("Start:"), 0, 0)
        g.addWidget(self.spin_cut_start, 0, 1)
        g.addWidget(QLabel("End:"), 1, 0)
        g.addWidget(self.spin_cut_end, 1, 1)
        v.addLayout(g)
        self.btn_set_loop = QPushButton("🔁 Set Loop to Selection")
        self.btn_set_loop.setMinimumHeight(40)
        self.btn_set_loop.clicked.connect(self._req_set_loop)
        self.btn_cut = QPushButton("✂️ Cut Selected Area")
        self.btn_cut.setMinimumHeight(40)
        self.btn_cut.setStyleSheet("QPushButton { background-color: #aa3333; color: white; font-weight: bold; } QPushButton:hover { background-color: #cc4444; }")
        self.btn_cut.clicked.connect(self._apply_crop)
        v.addWidget(self.btn_set_loop)
        v.addWidget(self.btn_cut)
        layout.addWidget(grp_sel)
        grp_speed = QFrame()
        grp_speed.setStyleSheet("background: #383838; border-radius: 6px; padding: 10px; margin-top: 10px;")
        sv = QVBoxLayout(grp_speed)
        sv.addWidget(QLabel("⏩ Time Stretch"))
        sh = QHBoxLayout()
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.5, 2.0)
        self.spin_speed.setSingleStep(0.1)
        self.spin_speed.setValue(1.0)
        self.spin_speed.setSuffix(" x")
        self.spin_speed.setMinimumHeight(35)
        self.spin_speed.setStyleSheet("QDoubleSpinBox { font-size: 14px; background: #555; color: white; border: 1px solid #777; } QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 25px; }")
        self.btn_apply_speed = QPushButton("Apply")
        self.btn_apply_speed.setMinimumHeight(35)
        self.btn_apply_speed.clicked.connect(self._apply_speed_change)
        sh.addWidget(self.spin_speed)
        sh.addWidget(self.btn_apply_speed)
        sv.addLayout(sh)
        layout.addWidget(grp_speed)
        layout.addStretch()
    def set_track(self, track: Optional[Track]):
        self.current_track = track
        self.refresh_from_ctx()
    def refresh_from_ctx(self):
        self.sld_master_vol.setValue(int(self.ctx.master_volume * 100))
        self.dial_master.setAngle(self.ctx.master_angle_deg)
        t = self.current_track
        if not t:
            self.grp_track.setEnabled(False)
            self.tab_fx.setEnabled(False)
            self.tab_edit.setEnabled(False)
            self.lbl_track_name.setText("No Track Selected")
            return
        self.grp_track.setEnabled(True)
        self.tab_fx.setEnabled(True)
        self.tab_edit.setEnabled(True)
        self.lbl_track_name.setText(t.name)
        self.sld_track_vol.blockSignals(True)
        self.sld_track_vol.setValue(int(t.volume * 100))
        self.sld_track_vol.blockSignals(False)
        self.dial_track.setAngle(t.angle_deg)
        self.sld_eq_low.blockSignals(True)
        self.sld_eq_mid.blockSignals(True)
        self.sld_eq_high.blockSignals(True)
        self.sld_eq_low.setValue(int(t.eq_low * 100))
        self.sld_eq_mid.setValue(int(t.eq_mid * 100))
        self.sld_eq_high.setValue(int(t.eq_high * 100))
        self.sld_eq_low.blockSignals(False)
        self.sld_eq_mid.blockSignals(False)
        self.sld_eq_high.blockSignals(False)
    def _req_set_loop(self):
        s = self.spin_cut_start.value()
        e = self.spin_cut_end.value()
        if e <= s:
            show_alert(self, "Invalid Range", "End time must be greater than Start time.", True)
            return
        self.loop_request.emit(s, e)
        show_alert(self, "Loop Set", f"Loop range set: {s:.2f}s ~ {e:.2f}s")
    def _apply_pitch_shift(self):
        if not self.current_track: return
        steps = self.spin_pitch.value()
        if steps == 0: return
        try:
            self.process_started.emit()
            QApplication.processEvents()
            y = self.current_track.data.T
            y_shift_l = librosa.effects.pitch_shift(y[0], sr=self.current_track.sr, n_steps=steps)
            y_shift_r = librosa.effects.pitch_shift(y[1], sr=self.current_track.sr, n_steps=steps)
            new_data = np.stack([y_shift_l, y_shift_r], axis=1)
            self.current_track.update_data(new_data)
            self.track_modified.emit()
            show_alert(self, "Success", "Applied!")
        except Exception as e:
            show_alert(self, "Error", str(e), True)
        finally:
            self.process_finished.emit()
    def _apply_speed_change(self):
        if not self.current_track: return
        rate = self.spin_speed.value()
        if rate == 1.0: return
        try:
            self.process_started.emit()
            QApplication.processEvents()
            y = self.current_track.data.T
            y_str_l = librosa.effects.time_stretch(y[0], rate=rate)
            y_str_r = librosa.effects.time_stretch(y[1], rate=rate)
            new_data = np.stack([y_str_l, y_str_r], axis=1)
            self.current_track.update_data(new_data)
            self.track_modified.emit()
            show_alert(self, "Success", "Applied!")
        except Exception as e:
            show_alert(self, "Error", str(e), True)
        finally:
            self.process_finished.emit()
    def _apply_crop(self):
        if not self.current_track: return
        s = self.spin_cut_start.value()
        e = self.spin_cut_end.value()
        if s >= e:
            show_alert(self, "Error", "Invalid range.", True)
            return
        sr = self.current_track.sr
        start_idx = int(s * sr)
        end_idx = int(e * sr)
        if start_idx >= len(self.current_track.data): return
        new_data = self.current_track.data[start_idx:end_idx]
        self.current_track.update_data(new_data)
        self.track_modified.emit()
        show_alert(self, "Success", "Track cropped to selection!")

class StudioMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Mixing Studio (v31 - Final Complete Fix)")
        self.resize(1400, 950)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; }
            QScrollArea, QFrame { border: none; }
            QLabel { color: #ffffff; }
            QPushButton { 
                background-color: #444; color: #ffffff; border: 1px solid #666; 
                padding: 6px; border-radius: 4px; font-size: 13px;
            }
            QPushButton:hover { background-color: #555; }
            QPushButton:pressed { background-color: #333; }
            QSlider::groove:horizontal { background: #444; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #00aaff; width: 14px; margin: -5px 0; border-radius: 7px; }
            QCheckBox { color: #ffffff; spacing: 5px; }
            QProgressBar { text-align: center; color: white; border: 1px solid #555; }
            QDockWidget { color: white; titlebar-close-icon: url(close.png); titlebar-normal-icon: url(undock.png); }
            QDockWidget::title { background: #333; text-align: left; padding-left: 5px; }
        """)
        try:
            self.logger = init_logging()
            cache_dir = ensure_writable_cache()
            apply_env_for_cache(cache_dir)
        except Exception as e:
            print(f"Init Error: {e}")
        self.ctx = ProjectContext()
        self.engine = AudioEngine(self.ctx)
        self.worker = None
        self.loader = None
        self.track_widgets = []
        self.selected_track_id: Optional[str] = None
        self.input_media_path: Optional[str] = None
        
        self.visualizer: Optional[VisualizerScreen] = None
        self.is_seeking = False
        self.dock_vis = None
        self.dock_insp = None
        
        self._init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_playback_ui)
        self.timer.start(33)

    def _init_ui(self):
        self.setDockNestingEnabled(True)

        menubar = self.menuBar()
        menubar.setStyleSheet("QMenuBar { background-color: #333; color: white; } QMenu { background-color: #333; color: white; }")
        
        file_menu = menubar.addMenu("File")
        action_change_bg = QAction("Set Background Image", self)
        action_change_bg.triggered.connect(self.change_background_image)
        file_menu.addAction(action_change_bg)
        
        view_menu = menubar.addMenu("View")
        act_layout_def = QAction("Layout: Default", self)
        act_layout_def.triggered.connect(lambda: self.apply_layout("default"))
        view_menu.addAction(act_layout_def)
        act_layout_wide = QAction("Layout: Widescreen", self)
        act_layout_wide.triggered.connect(lambda: self.apply_layout("wide"))
        view_menu.addAction(act_layout_wide)
        act_layout_audio = QAction("Layout: Focus Audio", self)
        act_layout_audio.triggered.connect(lambda: self.apply_layout("audio"))
        view_menu.addAction(act_layout_audio)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(5, 5, 5, 5)
        
        top_bar = QHBoxLayout()
        btn_open = QPushButton("📂 New Project")
        btn_open.clicked.connect(self.load_audio_dialog)
        btn_add = QPushButton("➕ Add Layer")
        btn_add.clicked.connect(self.add_layer_dialog)
        btn_bg = QPushButton("🖼 Set BG Image")
        btn_bg.clicked.connect(self.change_background_image)
        btn_vis_settings = QPushButton("⚙ Vis Settings")
        btn_vis_settings.clicked.connect(self.open_vis_settings)
        self.btn_spec = QPushButton("📊 Spectrum: OFF")
        self.btn_spec.setCheckable(True)
        self.btn_spec.setChecked(False) 
        self.btn_spec.clicked.connect(self.on_toggle_spectrum)
        btn_export = QPushButton("💾 Export")
        btn_export.clicked.connect(self.export_media)
        self.chk_gpu = QCheckBox("GPU")
        self.chk_gpu.setChecked(True)
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #aaa;")

        top_bar.addWidget(btn_open)
        top_bar.addWidget(btn_add)
        top_bar.addWidget(btn_bg)
        top_bar.addWidget(btn_vis_settings)
        top_bar.addWidget(self.btn_spec)
        top_bar.addWidget(btn_export)
        top_bar.addWidget(self.chk_gpu)
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()
        central_layout.addLayout(top_bar)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        central_layout.addWidget(self.progress)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.tracks_container = QWidget()
        self.tracks_container.setStyleSheet("background: #222;")
        self.tracks_layout = QVBoxLayout(self.tracks_container)
        self.tracks_layout.addStretch(1)
        scroll.setWidget(self.tracks_container)
        central_layout.addWidget(scroll, 1)

        btm = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setFixedSize(100, 45)
        self.btn_play.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #0077aa;")
        self.btn_play.clicked.connect(self.toggle_play)
        btn_rw = QPushButton("⏪ -5s")
        btn_rw.setFixedSize(60, 45)
        btn_rw.clicked.connect(lambda: self.seek_delta(-5))
        btn_ff = QPushButton("⏩ +5s")
        btn_ff.setFixedSize(60, 45)
        btn_ff.clicked.connect(lambda: self.seek_delta(5))
        self.btn_loop = QPushButton("🔁 Loop: OFF")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setFixedSize(90, 45)
        self.btn_loop.clicked.connect(self.toggle_loop)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setFixedWidth(100)
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_time.setStyleSheet("font-family: Consolas; font-size: 14px;")
        self.slider_seek = QSlider(Qt.Orientation.Horizontal)
        self.slider_seek.setRange(0, 1000)
        self.slider_seek.sliderPressed.connect(self.on_seek_press)
        self.slider_seek.sliderReleased.connect(self.on_seek_release)
        btm.addWidget(self.btn_play)
        btm.addWidget(btn_rw)
        btm.addWidget(btn_ff)
        btm.addWidget(self.btn_loop)
        btm.addWidget(self.lbl_time)
        btm.addWidget(self.slider_seek, 1)
        central_layout.addLayout(btm)

        # -- Unified Visualizer --
        self.visualizer = VisualizerScreen()
        
        self.video_player = QMediaPlayer(self)
        self.video_audio = QAudioOutput(self)
        self.video_audio.setVolume(0)
        self.video_player.setAudioOutput(self.video_audio)
        self.video_player.setVideoOutput(self.visualizer.video_sink)
        
        self.dock_vis = QDockWidget("Visualizer / Video", self)
        self.dock_vis.setWidget(self.visualizer)
        self.dock_vis.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.dock_vis)

        # -- Dock 2: Inspector --
        self.inspector = InspectorPanel(self.ctx)
        self.inspector.track_modified.connect(self._on_track_data_changed)
        self.inspector.loop_request.connect(self._on_loop_request)
        self.inspector.process_started.connect(self._on_process_start)
        self.inspector.process_finished.connect(self._on_process_finish)
        
        self.dock_insp = QDockWidget("Inspector", self)
        self.dock_insp.setWidget(self.inspector)
        self.dock_insp.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_insp)

    def apply_layout(self, layout_type):
        if layout_type == "default":
            self.dock_vis.setVisible(True)
            self.dock_insp.setVisible(True)
            self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.dock_vis)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_insp)
            self.dock_vis.setFloating(False)
            self.dock_insp.setFloating(False)
        elif layout_type == "wide":
            self.dock_vis.setVisible(True)
            self.dock_insp.setVisible(True)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_vis)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_insp)
        elif layout_type == "audio":
            self.dock_vis.setVisible(False)
            self.dock_insp.setVisible(True)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_insp)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected_track()
        else:
            super().keyPressEvent(event)

    def delete_selected_track(self):
        if not self.selected_track_id: return
        self.ctx.remove_track(self.selected_track_id)
        widget_to_remove = None
        for w in self.track_widgets:
            if w.track.id == self.selected_track_id:
                widget_to_remove = w
                break
        if widget_to_remove:
            self.tracks_layout.removeWidget(widget_to_remove)
            widget_to_remove.deleteLater()
            self.track_widgets.remove(widget_to_remove)
        self.selected_track_id = None
        self.inspector.set_track(None)
        self._on_track_data_changed()
        show_alert(self, "Deleted", "Track removed.")

    def _on_process_start(self):
        if self.ctx.is_playing: self.toggle_play() 
        self.setEnabled(False) 
        self.lbl_status.setText("Processing... Please wait.")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor) 

    def _on_process_finish(self):
        self.setEnabled(True)
        self.lbl_status.setText("Ready")
        QApplication.restoreOverrideCursor()

    def _on_loop_request(self, start_sec, end_sec):
        self.ctx.loop_start = int(start_sec * self.ctx.sample_rate)
        self.ctx.loop_end = int(end_sec * self.ctx.sample_rate)
        if not self.btn_loop.isChecked(): self.btn_loop.click() 
        else: self.ctx.is_looping = True
        self.btn_loop.setText("🔁 Loop: ON")
        self.btn_loop.setStyleSheet("background-color: #00ccff; color: black; font-weight: bold;")

    def toggle_play(self):
        if self.ctx.is_playing:
            self.engine.stop()
            self.btn_play.setText("▶ Play")
            if self.video_player: self.video_player.pause()
        else:
            if self.ctx.total_frames <= 0: return
            self.engine.start()
            self.btn_play.setText("⏸ Pause")
            if self.video_player: self.video_player.play()

    def seek_delta(self, sec):
        if self.ctx.total_frames <= 0: return
        cur_sec = self.ctx.current_frame / self.ctx.sample_rate
        new_frame = int((cur_sec + sec) * self.ctx.sample_rate)
        self.engine.seek(new_frame)
        self.update_playback_ui()

    def toggle_loop(self):
        is_loop = self.btn_loop.isChecked()
        self.ctx.is_looping = is_loop
        if is_loop:
            self.btn_loop.setText("🔁 Loop: ON")
            self.btn_loop.setStyleSheet("background-color: #00ccff; color: black; font-weight: bold;")
            if self.ctx.loop_end <= self.ctx.loop_start:
                self.ctx.loop_start = 0
                self.ctx.loop_end = self.ctx.total_frames
        else:
            self.btn_loop.setText("🔁 Loop: OFF")
            self.btn_loop.setStyleSheet("")

    def change_background_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Background Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.visualizer.set_background_image(path)
            self.video_player.stop()
            show_alert(self, "Background Set", f"Image set: {os.path.basename(path)}")

    def open_vis_settings(self):
        dlg = VisualizerSettingsDialog(self, self.visualizer.get_settings())
        if dlg.exec():
            self.visualizer.set_settings(dlg.get_settings())

    def on_toggle_spectrum(self, checked):
        self.visualizer.toggle_spectrum(checked)
        if checked:
            self.btn_spec.setText("📊 Spectrum: ON")
            self.btn_spec.setStyleSheet("background-color: #00aa00; color: white;")
        else:
            self.btn_spec.setText("📊 Spectrum: OFF")
            self.btn_spec.setStyleSheet("")

    def load_audio_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "New Project", "", "Media (*.mp3 *.wav *.mp4 *.mkv)")
        if path: self.start_ai_process(path)

    def add_layer_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add Layer", "", "Media (*.mp3 *.wav)")
        if path: self.start_external_load(path)

    def start_ai_process(self, path):
        self.input_media_path = path
        self.engine.stop()
        self.ctx.clear()
        self._clear_track_ui()
        self.visualizer.clear_background()
        
        if is_video_file(path):
            self.lbl_status.setText("영상 로드됨")
            self.video_player.setSource(QUrl.fromLocalFile(path))
            self.video_player.pause() 
        else:
            self.lbl_status.setText("오디오 로드됨")
            self.video_player.setSource(QUrl())
        
        dev = "cuda" if self.chk_gpu.isChecked() else "cpu"
        self.worker = AISeparationWorker(path, user_sr=self.ctx.sample_rate, device_pref=dev)
        self.worker.progress.connect(lambda v, m: (self.progress.setValue(v), self.lbl_status.setText(m)))
        self.worker.finished.connect(self.on_tracks_ready)
        self.worker.failed.connect(lambda e: show_alert(self, "Error", e, True))
        self.progress.setVisible(True)
        self.worker.start()

    def on_tracks_ready(self, tracks):
        self.progress.setVisible(False)
        self.lbl_status.setText("Ready")
        for t in tracks:
            self.ctx.add_track(t)
            self.add_track_widget(t)
        if self.ctx.tracks: self._select_track(self.ctx.tracks[0].id)
        show_alert(self, "Complete", "AI Separation Done!")

    def start_external_load(self, path):
        self.loader = ExternalFileLoader(path, sr=self.ctx.sample_rate)
        self.loader.finished.connect(self.on_single_track_ready)
        self.loader.start()

    def on_single_track_ready(self, t):
        self.ctx.add_track(t)
        self.add_track_widget(t)
        self._select_track(t.id)

    def add_track_widget(self, t):
        w = TrackStrip(t)
        w.clicked.connect(self._select_track)
        w.rename_requested.connect(self._on_rename)
        w.selection_changed.connect(self.inspector.update_selection)
        self.track_widgets.append(w)
        self.tracks_layout.insertWidget(self.tracks_layout.count()-1, w)

    def _select_track(self, tid):
        self.selected_track_id = tid
        track = next((t for t in self.ctx.tracks if t.id == tid), None)
        for w in self.track_widgets: w.set_selected(w.track.id == tid)
        self.inspector.set_track(track)
        if track: self.inspector.tabs.setCurrentIndex(0) 

    def _on_rename(self, tid, new_name):
        if self.inspector.current_track and self.inspector.current_track.id == tid:
            self.inspector.lbl_track_name.setText(new_name)

    def _on_track_data_changed(self):
        self.ctx.update_total_frames()
        self.inspector.refresh_from_ctx()
        for w in self.track_widgets:
            if w.track == self.inspector.current_track:
                w.wave.vis_min = w.track.vis_min
                w.wave.vis_max = w.track.vis_max
                w.wave.update()

    def _clear_track_ui(self):
        self.track_widgets = []
        while self.tracks_layout.count() > 0:
            item = self.tracks_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.tracks_layout.addStretch(1)
        self.inspector.set_track(None)

    def on_seek_press(self): self.is_seeking = True
    def on_seek_release(self):
        if self.ctx.total_frames > 0:
            self.engine.seek(int(self.slider_seek.value()/1000 * self.ctx.total_frames))
        self.is_seeking = False

    def update_playback_ui(self):
        if self.ctx.is_playing and self.engine.vis_buffer is not None:
            self.visualizer.set_audio_data(self.engine.vis_buffer)
        else:
            self.visualizer.set_audio_data(None)

        if self.ctx.total_frames > 0:
            cur = self.ctx.current_frame
            tot = self.ctx.total_frames
            self.lbl_time.setText(f"{cur//44100//60:02}:{cur//44100%60:02} / {tot//44100//60:02}:{tot//44100%60:02}")
            ratio = (cur / tot) if tot > 0 else 0
            if not self.is_seeking: self.slider_seek.setValue(int(ratio * 1000))
            for w in self.track_widgets: w.set_playhead_ratio(ratio)
            
            if self.ctx.is_playing and self.video_player.source().isValid():
                v_pos = self.video_player.position()
                a_pos = int(cur / self.ctx.sample_rate * 1000)
                diff = abs(v_pos - a_pos)
                if diff > 100: 
                    self.video_player.setPosition(a_pos)
                if self.video_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                    self.video_player.play()

    def export_media(self):
        if not self.ctx.tracks: return
        filters = "Audio Only (*.wav)"
        path, _ = QFileDialog.getSaveFileName(self, "Export", "mix.wav", filters)
        if not path: return

        if self.engine.export_mix_to_wav(path):
            show_alert(self, "Success", f"Exported: {path}")
        else:
            show_alert(self, "Error", "Export Failed", True)

    def closeEvent(self, e):
        self.engine.stop()
        if self.worker and self.worker.isRunning(): self.worker.quit()
        super().closeEvent(e)

def main():
    app = QApplication(sys.argv)
    win = StudioMainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()