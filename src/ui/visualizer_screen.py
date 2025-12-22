import math
import numpy as np
from PyQt6.QtWidgets import (QWidget, QDialog, QVBoxLayout, QTabWidget, QFormLayout, 
                             QSlider, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, 
                             QCheckBox, QLabel, QGroupBox, QFontComboBox, 
                             QColorDialog, QDialogButtonBox, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPoint, QPointF, QRect
from PyQt6.QtGui import (QPainter, QColor, QPen, QBrush, QFont, QFontMetrics, 
                         QPixmap, QImage, QPolygonF, QLinearGradient)
from PyQt6.QtMultimedia import QVideoSink, QVideoFrame

# --- Settings Dialog ---
class VisualizerSettingsDialog(QDialog):
    textAdded = pyqtSignal(str, float)

    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Visualizer Detailed Settings")
        self.resize(500, 800)
        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: white; }
            QLabel, QCheckBox { color: white; }
            QPushButton { background: #444; color: white; padding: 5px; border: 1px solid #666; }
            QSpinBox, QDoubleSpinBox, QComboBox, QFontComboBox, QLineEdit { background: #333; color: white; border: 1px solid #555; }
            QGroupBox { border: 1px solid #555; margin-top: 10px; padding-top: 15px; }
        """)
        
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab_spectrum = QWidget()
        self._init_spectrum_tab()
        self.tabs.addTab(self.tab_spectrum, "Spectrum")

        self.tab_text = QWidget()
        self._init_text_tab()
        self.tabs.addTab(self.tab_text, "Text Creation & Style")

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        main_layout.addWidget(self.buttons)

        if current_settings:
            self._load_settings(current_settings)

    def _init_spectrum_tab(self):
        layout = QFormLayout(self.tab_spectrum)
        self.sld_opacity = QSlider(Qt.Orientation.Horizontal); self.sld_opacity.setRange(0, 100)
        self.combo_shape = QComboBox(); self.combo_shape.addItems(["Linear", "Circular", "Line"])
        self.combo_style = QComboBox(); self.combo_style.addItems(["Bar", "Line"])
        self.spin_thick = QSpinBox(); self.spin_thick.setRange(1, 100)
        self.spin_scale = QDoubleSpinBox(); self.spin_scale.setRange(0.1, 20.0); self.spin_scale.setSingleStep(0.1)
        self.spin_height = QDoubleSpinBox(); self.spin_height.setRange(0.1, 100.0)
        self.spin_bar_count = QSpinBox(); self.spin_bar_count.setRange(10, 256); self.spin_bar_count.setValue(60)
        self.spin_radius = QDoubleSpinBox(); self.spin_radius.setRange(0.0, 1.0); self.spin_radius.setSingleStep(0.05); self.spin_radius.setValue(0.3)
        
        self.btn_color_bot = QPushButton("Start Color"); self.btn_color_bot.clicked.connect(lambda: self.pick_color('bot'))
        self.btn_color_top = QPushButton("End Color"); self.btn_color_top.clicked.connect(lambda: self.pick_color('top'))
        self.btn_color_ov = QPushButton("Mask Color"); self.btn_color_ov.clicked.connect(lambda: self.pick_color('ov'))
        
        self.chk_mask = QCheckBox("Enable Mask Mode (투명 구멍)")
        self.col_bot = QColor(0,0,255); self.col_top = QColor(0,255,255); self.col_ov = QColor(0,0,0)

        layout.addRow("BG Opacity:", self.sld_opacity)
        layout.addRow("Bar Count:", self.spin_bar_count)
        layout.addRow("Shape:", self.combo_shape)
        layout.addRow("Radius (Circular):", self.spin_radius)
        layout.addRow("Draw Style:", self.combo_style)
        layout.addRow("Thickness:", self.spin_thick)
        layout.addRow("Sensitivity:", self.spin_scale)
        layout.addRow("Height Scale:", self.spin_height)
        layout.addRow(self.btn_color_bot, self.btn_color_top)
        layout.addRow("Mask Color:", self.btn_color_ov)
        layout.addRow(self.chk_mask)

    def _init_text_tab(self):
        layout = QVBoxLayout(self.tab_text)
        
        create_grp = QGroupBox("Create New Text")
        cl = QFormLayout(create_grp)
        self.txt_input = QLineEdit()
        self.spin_duration = QDoubleSpinBox(); self.spin_duration.setValue(5.0)
        self.btn_create_text = QPushButton("Create Text")
        self.btn_create_text.clicked.connect(self._on_create_text)
        cl.addRow("Content:", self.txt_input)
        cl.addRow("Duration:", self.spin_duration)
        cl.addRow(self.btn_create_text)
        layout.addWidget(create_grp)
        
        style_grp = QGroupBox("Global Text Style")
        sl = QFormLayout(style_grp)
        self.sld_text_op = QSlider(Qt.Orientation.Horizontal); self.sld_text_op.setRange(0, 100); self.sld_text_op.setValue(100)
        self.chk_text_mask = QCheckBox("Text Masking Mode")
        self.font_fam = QFontComboBox()
        self.spin_size = QSpinBox(); self.spin_size.setRange(10, 300)
        self.chk_bold = QCheckBox("Bold")
        self.chk_italic = QCheckBox("Italic")
        
        sl.addRow("Opacity:", self.sld_text_op)
        sl.addRow(self.chk_text_mask)
        sl.addRow("Font:", self.font_fam)
        sl.addRow("Size:", self.spin_size)
        sl.addRow(self.chk_bold, self.chk_italic)
        layout.addWidget(style_grp)
        layout.addStretch()

    def _on_create_text(self):
        text = self.txt_input.text()
        if text:
            self.textAdded.emit(text, self.spin_duration.value())
            self.txt_input.clear()

    def pick_color(self, mode):
        t = self.col_bot if mode=='bot' else (self.col_top if mode=='top' else self.col_ov)
        c = QColorDialog.getColor(t, self, "Pick Color")
        if c.isValid():
            if mode=='bot': self.col_bot = c
            elif mode=='top': self.col_top = c
            else: self.col_ov = c
            btn = self.btn_color_bot if mode=='bot' else (self.btn_color_top if mode=='top' else self.btn_color_ov)
            btn.setStyleSheet(f"background: {c.name()}; color: {'black' if c.lightness()>128 else 'white'};")

    def _load_settings(self, s):
        self.sld_opacity.setValue(int(s.get('bg_opacity', 1.0)*100))
        self.spin_bar_count.setValue(s.get('bar_count', 60))
        self.spin_radius.setValue(s.get('radius', 0.3))
        
        shape_idx = 0
        if s.get('shape') == 'circular': shape_idx = 1
        elif s.get('shape') == 'line': shape_idx = 2
        self.combo_shape.setCurrentIndex(shape_idx)
        
        self.spin_scale.setValue(s.get('scale', 0.1))
        self.spin_height.setValue(s.get('height_scale', 1.0))
        self.spin_thick.setValue(s.get('thickness', 5))
        self.chk_mask.setChecked(s.get('mask_mode', False))
        
        self.col_bot = s.get('color_bot', QColor(0,0,255))
        self.col_top = s.get('color_top', QColor(0,255,255))
        self.col_ov = s.get('color_overlay', QColor(0,0,0))
        
        for mode, col in [('bot', self.col_bot), ('top', self.col_top), ('ov', self.col_ov)]:
            btn = self.btn_color_bot if mode=='bot' else (self.btn_color_top if mode=='top' else self.btn_color_ov)
            btn.setStyleSheet(f"background: {col.name()}; color: {'black' if col.lightness()>128 else 'white'};")

        self.sld_text_op.setValue(int(s.get('text_opacity', 1.0)*100))
        self.chk_text_mask.setChecked(s.get('text_mask_mode', False))
        
        sub = s.get('sub_cfg', {})
        self.font_fam.setCurrentFont(QFont(sub.get('font', 'Arial')))
        self.spin_size.setValue(sub.get('size', 30))
        self.chk_bold.setChecked(sub.get('bold', False))
        self.chk_italic.setChecked(sub.get('italic', False))

    def get_settings(self):
        idx = self.combo_shape.currentIndex()
        shape = 'linear'
        if idx == 1: shape = 'circular'
        elif idx == 2: shape = 'line'
        
        return {
            'bg_opacity': self.sld_opacity.value()/100,
            'bar_count': self.spin_bar_count.value(),
            'shape': shape,
            'radius': self.spin_radius.value(),
            'draw_mode': 'bar',
            'thickness': self.spin_thick.value(),
            'scale': self.spin_scale.value(),
            'height_scale': self.spin_height.value(),
            'mask_mode': self.chk_mask.isChecked(),
            'color_bot': self.col_bot, 'color_top': self.col_top, 'color_overlay': self.col_ov,
            'text_opacity': self.sld_text_op.value()/100,
            'text_mask_mode': self.chk_text_mask.isChecked(),
            'sub_cfg': {
                'font': self.font_fam.currentFont().family(),
                'size': self.spin_size.value(),
                'bold': self.chk_bold.isChecked(),
                'italic': self.chk_italic.isChecked()
            }
        }

# --- Main Screen ---
class VisualizerScreen(QWidget):
    settingsChanged = pyqtSignal(dict) 
    settingsRequested = pyqtSignal()

    def __init__(self, ctx=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx # Need context to access subtitles
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setMinimumHeight(300)
        
        self.video_sink = QVideoSink()
        self.video_sink.videoFrameChanged.connect(self._on_video_frame)
        self.current_video_frame = None
        self.audio_data = None
        self.bar_count = 60
        self.decay = np.zeros(self.bar_count)
        self.bg_pixmap = None
        self.use_bg_image = False
        
        # Init Settings
        self.show_spectrum = True 
        self.bg_opacity = 1.0
        self.shape = 'linear'
        self.radius = 0.3
        self.draw_mode = 'bar'
        self.thickness = 5
        self.scale = 0.1 
        self.height_scale = 1.0
        self.color_bot = QColor(0,0,255); self.color_top = QColor(0,255,255); self.color_overlay = QColor(0,0,0)
        self.mask_mode = False 
        self.text_opacity = 1.0
        self.text_mask_mode = False
        self.sub_cfg = {'text':'', 'size':30, 'font':'Arial', 'bold':False, 'italic':False, 'x':50, 'y':100}
        
        self._dragging_target = None
        self._drag_start_pos = QPoint()
        self._drag_start_elem_pos = QPoint()
        
        self.btn_settings = QPushButton("⚙ Settings", self)
        self.btn_settings.setGeometry(self.width()-90, 10, 80, 30)
        self.btn_settings.setStyleSheet("background: rgba(0,0,0,150); color: white; border: 1px solid white; border-radius: 4px;")
        self.btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_settings.clicked.connect(self.settingsRequested.emit)

    def resizeEvent(self, e):
        self.btn_settings.move(self.width()-90, 10)
        super().resizeEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            pos = e.position().toPoint()
            if self._hit_test(pos):
                self._dragging_target = 'sub'; self._drag_start_pos = pos
                self._drag_start_elem_pos = QPoint(self.sub_cfg['x'], self.sub_cfg['y'])
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._dragging_target == 'sub':
            delta = e.position().toPoint() - self._drag_start_pos
            self.sub_cfg['x'] = self._drag_start_elem_pos.x() + delta.x()
            self.sub_cfg['y'] = self._drag_start_elem_pos.y() + delta.y()
            self.update()

    def mouseReleaseEvent(self, e):
        self._dragging_target = None; self.setCursor(Qt.CursorShape.ArrowCursor)

    def _hit_test(self, pos):
        return bool(self.sub_cfg.get('text'))

    def _on_video_frame(self, frame: QVideoFrame):
        if frame.isValid():
            self.current_video_frame = frame.toImage()
            self.update()

    def set_settings(self, s):
        self.bg_opacity = s.get('bg_opacity', 1.0)
        self.bar_count = s.get('bar_count', 60)
        self.shape = s.get('shape', 'linear')
        self.radius = s.get('radius', 0.3)
        self.draw_mode = s.get('draw_mode', 'bar')
        self.thickness = s.get('thickness', 5)
        self.scale = s.get('scale', 0.1)
        self.height_scale = s.get('height_scale', 1.0)
        self.mask_mode = s.get('mask_mode', False)
        self.color_bot = s.get('color_bot')
        self.color_top = s.get('color_top')
        self.color_overlay = s.get('color_overlay')
        self.text_opacity = s.get('text_opacity', 1.0)
        self.text_mask_mode = s.get('text_mask_mode', False)
        self.sub_cfg.update(s.get('sub_cfg', {}))
        
        if len(self.decay) != self.bar_count: self.decay = np.zeros(self.bar_count)
        self.update()

    def get_settings(self):
        return {
            'bg_opacity': self.bg_opacity, 'bar_count': self.bar_count, 'shape': self.shape, 'radius': self.radius,
            'draw_mode': self.draw_mode, 'thickness': self.thickness, 'scale': self.scale,
            'height_scale': self.height_scale, 'mask_mode': self.mask_mode,
            'color_bot': self.color_bot, 'color_top': self.color_top, 'color_overlay': self.color_overlay,
            'text_opacity': self.text_opacity, 'text_mask_mode': self.text_mask_mode,
            'sub_cfg': self.sub_cfg
        }

    def set_audio_data(self, data: np.ndarray):
        if data is None: self.audio_data = None
        else: self.audio_data = np.mean(data, axis=1)
        self.update()

    def set_background_image(self, path: str):
        if not path: self.bg_pixmap = None; self.use_bg_image = False
        else: self.bg_pixmap = QPixmap(path); self.use_bg_image = True; self.current_video_frame = None
        self.update()

    def clear_background(self):
        self.use_bg_image = False; self.current_video_frame = None; self.update()

    def _process_fft(self):
        if self.audio_data is None or len(self.audio_data) < 100: return np.zeros(self.bar_count)
        windowed = self.audio_data * np.hanning(len(self.audio_data))
        fft = np.fft.rfft(windowed)
        mag = np.abs(fft) * 100.0
        if len(mag) > self.bar_count:
            step = len(mag) // self.bar_count
            binned = np.array([np.mean(mag[i*step:(i+1)*step]) for i in range(self.bar_count)])
        else: binned = np.zeros(self.bar_count)
        binned = np.log10(binned + 1.0) * 300.0 * self.scale
        self.decay = np.maximum(self.decay * 0.80, binned)
        return self.decay

    def _get_draw_rect(self, w, h):
        img = self.bg_pixmap if (self.use_bg_image and self.bg_pixmap) else self.current_video_frame
        if img:
            img_ratio = img.width() / img.height()
            widget_ratio = w / h
            if img_ratio > widget_ratio:
                new_h = w / img_ratio
                return QRectF(0, (h - new_h) / 2, w, new_h)
            else:
                new_w = h * img_ratio
                return QRectF((w - new_w) / 2, 0, new_w, h)
        return QRectF(0, 0, w, h)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        p.fillRect(0, 0, w, h, QColor(20, 20, 20))

        draw_rect = self._get_draw_rect(w, h)
        
        if not self.use_bg_image and self.current_video_frame is None:
            p.setPen(QPen(QColor(50, 50, 50), 1, Qt.PenStyle.DotLine))
            for x in range(0, w, 50): p.drawLine(x, 0, x, h)
            for y in range(0, h, 50): p.drawLine(0, y, w, y)

        if self.use_bg_image and self.bg_pixmap:
            p.setOpacity(self.bg_opacity)
            p.drawPixmap(draw_rect.toRect(), self.bg_pixmap)
            p.setOpacity(1.0)
        elif self.current_video_frame:
            p.setOpacity(self.bg_opacity)
            p.drawImage(draw_rect.toRect(), self.current_video_frame)
            p.setOpacity(1.0)

        need_masking = self.mask_mode or self.text_mask_mode
        
        if need_masking:
            buffer = QPixmap(w, h)
            buffer.fill(Qt.GlobalColor.transparent)
            pb = QPainter(buffer)
            pb.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            if self.mask_mode: 
                pb.fillRect(draw_rect, self.color_overlay)
            
            if self.show_spectrum:
                decay_data = self._process_fft()
                op = QPainter.CompositionMode.CompositionMode_DestinationOut if self.mask_mode else QPainter.CompositionMode.CompositionMode_SourceOver
                pb.setCompositionMode(op)
                self._draw_spectrum_impl(pb, draw_rect, decay_data, self.mask_mode)
                pb.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            if self.text_mask_mode:
                pb.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
                self._draw_text(pb, self.sub_cfg, True, draw_rect)
                pb.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            else:
                self._draw_text(pb, self.sub_cfg, False, draw_rect)
            
            pb.end()
            p.drawPixmap(0, 0, buffer)
        else:
            if self.show_spectrum:
                decay_data = self._process_fft()
                self._draw_spectrum_impl(p, draw_rect, decay_data, False)
            self._draw_text(p, self.sub_cfg, False, draw_rect)

        # [NEW] Subtitles Draw Call
        if self.ctx:
            ct = self.ctx.current_frame / self.ctx.sample_rate
            self._draw_subtitles(p, w, h, ct)

        p.end()

    def _draw_spectrum_impl(self, p, rect, data, is_mask):
        if is_mask:
            p.setBrush(QBrush(QColor(255, 255, 255)))
            p.setPen(Qt.PenStyle.NoPen)
        else:
            grad = QLinearGradient(rect.left(), rect.bottom(), rect.left(), rect.top())
            grad.setColorAt(0, self.color_bot); grad.setColorAt(1, self.color_top)
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)

        cnt = len(data)
        w, h = rect.width(), rect.height()
        x0, y0 = rect.x(), rect.y()
        
        if self.shape == 'circular':
            cx, cy = x0 + w/2, y0 + h/2
            radius = min(w, h) * self.radius
            angle_step = 2 * math.pi / cnt
            
            pen = QPen(self.color_top, self.thickness)
            if is_mask: pen.setColor(Qt.GlobalColor.white)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            
            for i in range(cnt):
                val = data[i] * self.height_scale * 2.0
                angle = i * angle_step - (math.pi / 2)
                ox = cx + (radius + val) * math.cos(angle)
                oy = cy + (radius + val) * math.sin(angle)
                ix = cx + radius * math.cos(angle)
                iy = cy + radius * math.sin(angle)
                p.drawLine(QPointF(ix, iy), QPointF(ox, oy))
                
        elif self.shape == 'line':
            path = QPainterPath()
            step_w = w / max(1, cnt-1)
            path.moveTo(x0, y0 + h - (data[0] * self.height_scale))
            for i in range(1, cnt):
                x = x0 + i * step_w
                y = y0 + h - (data[i] * self.height_scale)
                path.lineTo(x, y)
                
            p.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(self.color_top, self.thickness)
            if is_mask: pen.setColor(Qt.GlobalColor.white)
            p.setPen(pen)
            p.drawPath(path)
            
        else:
            rect_w = w * 0.8
            bw = rect_w / cnt
            start_x = x0 + (w - rect_w) / 2
            for i in range(cnt):
                bh = min(h, data[i] * self.height_scale)
                bar_rect = QRectF(start_x + i*bw, y0 + h - bh, bw-1, bh)
                p.drawRect(bar_rect)

    def _draw_text(self, p, cfg, is_mask, draw_rect):
        t = cfg.get('text', '')
        if not t: return
        f = QFont(cfg.get('font', 'Arial'), cfg.get('size', 30))
        f.setBold(cfg.get('bold', False)); f.setItalic(cfg.get('italic', False))
        p.setFont(f)
        
        if is_mask: p.setPen(QColor(255, 255, 255))
        else: 
            c = QColor('white'); c.setAlphaF(self.text_opacity)
            p.setPen(c)
        p.drawText(cfg.get('x', 50), cfg.get('y', 100), t)

    def _draw_subtitles(self, p, w, h, ct):
        if not self.ctx.subtitles: return
        cur = None
        for s in self.ctx.subtitles:
            if s.start_time <= ct <= s.end_time:
                cur = s; break
        
        if cur:
            f = QFont("Malgun Gothic", 36, QFont.Weight.Bold)
            p.setFont(f)
            r = QRect(0, h-120, w, 100)
            p.setPen(QColor(0,0,0,180))
            p.drawText(r.adjusted(2,2,2,2), Qt.AlignmentFlag.AlignCenter, cur.text)
            p.setPen(QColor(255,255,0))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, cur.text)