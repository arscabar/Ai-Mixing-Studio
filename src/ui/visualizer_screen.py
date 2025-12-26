import math
import numpy as np
from PyQt6.QtWidgets import (QWidget, QDialog, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPoint, QPointF
from PyQt6.QtGui import (QPainter, QColor, QPen, QBrush, QFont, QFontMetrics, 
                         QPixmap, QImage, QLinearGradient, QPainterPath)
from PyQt6.QtMultimedia import QVideoSink, QVideoFrame

# --- Settings Dialog (호환성 유지용) ---
class VisualizerSettingsDialog(QDialog):
    textAdded = pyqtSignal(str, float)
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        pass

# --- Main Visualizer Screen ---
class VisualizerScreen(QWidget):
    settingsChanged = pyqtSignal(dict) 
    settingsRequested = pyqtSignal()

    def __init__(self, ctx=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx 
        
        # 렌더링 설정
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMouseTracking(True)
        self.setMinimumHeight(350)
        
        # [영상] 비디오 싱크 (데이터 수신 전용)
        self.video_sink = QVideoSink()
        self.video_sink.videoFrameChanged.connect(self._on_video_frame)
        self.current_video_frame = None
        
        # [오디오] 데이터 처리 변수
        self.audio_buffer = None
        self.bar_count = 60
        self.fft_data = np.zeros(self.bar_count)
        self.decay = np.zeros(self.bar_count)
        
        self.bg_pixmap = None
        self.use_bg_image = False
        
        # [설정] 모든 속성 초기화
        self.settings = {
            "show_spectrum": True,      
            "bg_opacity": 1.0,
            
            # Geometry
            "shape": "linear",          # linear, circular, line
            "bar_count": 64,
            "bar_gap": 2,
            "inner_radius": 50,         # Circle 모드 내부 반지름
            "radius": 50,               # 호환성용
            
            # Physics
            "sensitivity": 150,
            "scale": 1.0,               # 호환성용
            "smoothing": 0.5,
            "log_scale": True,
            "min_freq": 20,
            "max_freq": 16000,
            "height_scale": 1.0,
            
            # Appearance
            "color_bot": QColor(0, 0, 255),
            "color_top": QColor(0, 255, 255),
            "color_overlay": QColor(0, 0, 0),
            "fill": True,
            "round_caps": False,
            "stroke_width": 2,
            "mask_mode": False,
            
            # Internal Control
            "spec_visible": False       # 기본값 False (타임라인 이벤트로만 켜짐)
        }
        
        self.sub_cfg = {
            'text': '', 'size': 36, 'font': 'Arial', 
            'bold': False, 'italic': False, 'color': '#FFFFFF',
            'x': -1, 'y': -1
        }
        
        # 인터랙션 상태
        self._dragging_target = None
        self._drag_start_pos = QPoint()
        self._elem_start_pos = QPoint()
        
        # 설정 버튼
        #self.btn_settings = QPushButton("⚙ Settings", self)
        #self.btn_settings.setGeometry(self.width()-90, 10, 80, 30)
        #self.btn_settings.setStyleSheet("background: rgba(0,0,0,150); color: white; border: 1px solid white; border-radius: 4px;")
        #self.btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        #self.btn_settings.clicked.connect(self.settingsRequested.emit)

    def resizeEvent(self, e):
        #self.btn_settings.move(self.width()-90, 10)
        # [수정 완료] self.video_sink.resize(...) 삭제됨
        super().resizeEvent(e)

    # [메서드] 메인 윈도우 호환성
    def set_settings(self, data: dict):
        self.update_settings(data)

    def update_settings(self, data: dict):
        # 키 매핑 및 업데이트
        if 'scale' in data: self.settings['sensitivity'] = float(data['scale']) * 100
        if 'sensitivity' in data: self.settings['scale'] = float(data['sensitivity']) / 150.0
        
        if 'color' in data:
            c = QColor(data['color'])
            self.settings['color_top'] = c
            self.settings['color_bot'] = c.darker(200)
            
        for k, v in data.items():
            if k in self.settings:
                self.settings[k] = v
        
        self.update()

    # [메서드] 오디오 처리 (FFT)
    def set_audio_data(self, data: np.ndarray):
        if data is None:
            self.audio_buffer = None
            self.decay *= 0.9 # 서서히 줄어듦
        else:
            self.audio_buffer = data
            self._compute_fft()
        self.update()

    def _compute_fft(self):
        if self.audio_buffer is None or len(self.audio_buffer) < 2: return

        # Mono Mix
        if self.audio_buffer.ndim > 1: samples = self.audio_buffer.mean(axis=1)
        else: samples = self.audio_buffer

        # Windowing
        n = len(samples)
        window = np.hanning(n)
        fft_raw = np.abs(np.fft.rfft(samples * window))
        
        # Binning
        bar_count = self.settings.get('bar_count', 64)
        if len(self.decay) != bar_count:
            self.decay = np.zeros(bar_count)
            
        freq_bins = len(fft_raw)
        new_fft = np.zeros(bar_count)
        
        # Log Scale or Linear
        if self.settings.get('log_scale', True):
            log_idxs = np.logspace(0, np.log10(freq_bins), num=bar_count + 1, dtype=int)
            log_idxs = np.unique(log_idxs)
            if len(log_idxs) < bar_count + 1:
                log_idxs = np.linspace(0, freq_bins, bar_count + 1, dtype=int)
            
            for i in range(len(log_idxs)-1):
                start, end = log_idxs[i], log_idxs[i+1]
                if end <= start: end = start + 1
                segment = fft_raw[start:end]
                new_fft[i] = np.mean(segment) if len(segment) > 0 else 0
        else:
            chunk = freq_bins // bar_count
            for i in range(bar_count):
                start = i * chunk
                end = start + chunk
                segment = fft_raw[start:end]
                new_fft[i] = np.mean(segment) if len(segment) > 0 else 0

        # Scaling & Smoothing
        sens = self.settings.get('scale', 1.0)
        height_scale = self.settings.get('height_scale', 1.0)
        target = np.log10(new_fft + 1.0) * 100.0 * sens * height_scale
        
        smooth = self.settings.get('smoothing', 0.5)
        self.decay = self.decay * smooth + target * (1 - smooth)

    # [메서드] 비디오/이미지 처리
    def _on_video_frame(self, frame: QVideoFrame):
        if frame.isValid():
            self.current_video_frame = frame.toImage()
            self.update()

    def set_background_image(self, path):
        if path: self.bg_pixmap = QImage(path); self.use_bg_image = True
        else: self.bg_pixmap = None; self.use_bg_image = False
        self.update()
        
    def clear_background(self):
        self.bg_pixmap = None; self.use_bg_image = False; self.update()
        
    def get_settings(self): return self.settings

    # [이벤트] 마우스 인터랙션 (자막 드래그)
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if self.sub_cfg['text'] and e.position().y() > self.height() * 0.7:
                self._dragging_target = 'sub'
                self._drag_start_pos = e.position().toPoint()
                self._elem_start_pos = QPoint(
                    self.sub_cfg.get('x', self.width()//2), 
                    self.sub_cfg.get('y', self.height()-50)
                )
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._dragging_target == 'sub':
            delta = e.position().toPoint() - self._drag_start_pos
            self.sub_cfg['x'] = self._elem_start_pos.x() + delta.x()
            self.sub_cfg['y'] = self._elem_start_pos.y() + delta.y()
            self.update()
        else:
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._dragging_target = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        self.settingsRequested.emit()

    # [렌더링] Paint Event
    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        # 1. 배경
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        # 이미지 / 비디오
        opacity = self.settings.get('bg_opacity', 1.0)
        draw_rect = self.rect() # 전체 화면 채우기
        
        if self.use_bg_image and self.bg_pixmap:
            painter.setOpacity(opacity)
            painter.drawImage(draw_rect, self.bg_pixmap)
            painter.setOpacity(1.0)
        elif self.current_video_frame:
            painter.setOpacity(opacity)
            painter.drawImage(draw_rect, self.current_video_frame)
            painter.setOpacity(1.0)

        # 2. 스펙트럼 (spec_visible이 True이고 show_spectrum이 True일 때만)
        if self.settings.get('spec_visible', False) and self.settings.get('show_spectrum', True):
            self._draw_spectrum_main(painter, w, h)

        # 3. 자막
        if self.sub_cfg.get('text'):
            self._draw_subtitle(painter, w, h)

        painter.end()

    def _draw_spectrum_main(self, p, w, h):
        """마스킹 모드 등을 고려하여 스펙트럼 그리기 분기"""
        mask_mode = self.settings.get('mask_mode', False)
        text_mask = self.settings.get('text_mask_mode', False)
        
        if mask_mode or text_mask:
            # 버퍼에 그려서 합성 (CompositionMode)
            buffer = QPixmap(w, h)
            buffer.fill(Qt.GlobalColor.transparent)
            pb = QPainter(buffer)
            pb.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # 마스크 모드면 배경색(Overlay)을 먼저 깔고 뚫음
            if mask_mode:
                pb.fillRect(0, 0, w, h, self.settings.get('color_overlay', QColor(0,0,0)))
                op = QPainter.CompositionMode.CompositionMode_DestinationOut
            else:
                op = QPainter.CompositionMode.CompositionMode_SourceOver
            
            pb.setCompositionMode(op)
            self._draw_spectrum_impl(pb, w, h, mask_mode)
            pb.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            
            pb.end()
            p.drawPixmap(0, 0, buffer)
        else:
            # 일반 그리기
            self._draw_spectrum_impl(p, w, h, False)

    def _draw_spectrum_impl(self, p, w, h, is_mask):
        data = self.decay
        cnt = len(data)
        if cnt == 0: return

        # 설정값 가져오기
        shape = self.settings.get('shape', 'linear')
        fill = self.settings.get('fill', True)
        round_caps = self.settings.get('round_caps', False)
        stroke = self.settings.get('stroke_width', 2)
        
        color_top = self.settings.get('color_top', QColor('cyan'))
        color_bot = self.settings.get('color_bot', QColor('blue'))
        
        # 브러시/펜 설정
        if is_mask:
            p.setBrush(QBrush(QColor(255, 255, 255))) # 마스크는 흰색으로 뚫음
            p.setPen(Qt.PenStyle.NoPen)
        else:
            if fill and shape != 'line':
                grad = QLinearGradient(0, h, 0, 0)
                grad.setColorAt(0, color_bot)
                grad.setColorAt(1, color_top)
                p.setBrush(QBrush(grad))
                p.setPen(Qt.PenStyle.NoPen)
            else:
                p.setBrush(Qt.BrushStyle.NoBrush)
                pen = QPen(color_top, stroke)
                if round_caps: pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)

        # 1. Linear Bar
        if shape == 'linear':
            gap = self.settings.get('bar_gap', 2)
            total_gap = gap * (cnt - 1)
            bar_w = (w - total_gap) / cnt
            if bar_w < 1: bar_w = 1
            
            for i in range(cnt):
                val = data[i]
                bar_h = min(h, val * (h / 20))
                x = i * (bar_w + gap)
                y = h - bar_h
                
                if fill:
                    if round_caps: p.drawRoundedRect(QRectF(x, y, bar_w, bar_h), bar_w/2, bar_w/2)
                    else: p.drawRect(QRectF(x, y, bar_w, bar_h))
                else:
                    path = QPainterPath()
                    path.moveTo(x, h); path.lineTo(x, y); path.lineTo(x+bar_w, y); path.lineTo(x+bar_w, h)
                    p.drawPath(path)

        # 2. Circular
        elif shape == 'circular' or shape == 'circle': 
            cx, cy = w / 2, h / 2
            r_val = self.settings.get('radius', 50)
            radius = r_val if r_val > 1 else min(w, h) * r_val
            angle_step = 2 * math.pi / cnt
            
            p.save()
            p.translate(cx, cy)
            for i in range(cnt):
                val = data[i] * self.settings.get('height_scale', 1.0) * 2.0
                p.save()
                p.rotate(math.degrees(i * angle_step))
                if fill:
                    circum = 2 * math.pi * radius
                    bw = max(1, (circum / cnt) - self.settings.get('bar_gap', 2))
                    if round_caps: p.drawRoundedRect(QRectF(-bw/2, -radius - val, bw, val), bw/2, bw/2)
                    else: p.drawRect(QRectF(-bw/2, -radius - val, bw, val))
                else:
                    p.drawLine(QPointF(0, -radius), QPointF(0, -radius - val))
                p.restore()
            p.restore()

        # 3. Line (Curve)
        elif shape == 'line':
            path = QPainterPath()
            step_w = w / max(1, cnt-1)
            path.moveTo(0, h - (data[0] * self.settings.get('height_scale', 1.0)))
            for i in range(1, cnt):
                x = i * step_w
                y = h - (data[i] * self.settings.get('height_scale', 1.0))
                prev_x = (i-1) * step_w
                prev_y = h - (data[i-1] * self.settings.get('height_scale', 1.0))
                ctrl1 = QPointF(prev_x + step_w/2, prev_y)
                ctrl2 = QPointF(x - step_w/2, y)
                path.cubicTo(ctrl1, ctrl2, QPointF(x, y))
            if fill:
                path.lineTo(w, h); path.lineTo(0, h); path.closeSubpath()
            p.drawPath(path)

    def _draw_subtitle(self, p, w, h):
        text = self.sub_cfg.get('text', '')
        if not text: return
        size = self.sub_cfg.get('size', 36)
        col_str = self.sub_cfg.get('color', '#FFFFFF')
        f = QFont(self.sub_cfg.get('font', 'Arial'), size)
        f.setBold(self.sub_cfg.get('bold', False))
        f.setItalic(self.sub_cfg.get('italic', False))
        p.setFont(f)
        
        sx = self.sub_cfg.get('x', -1)
        sy = self.sub_cfg.get('y', -1)
        fm = QFontMetrics(f)
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        
        if sx < 0 or sy < 0:
            sx = (w - tw) / 2
            sy = h - th - 50
            self.sub_cfg['x'] = sx
            self.sub_cfg['y'] = sy
            
        path = QPainterPath()
        path.addText(sx, sy + fm.ascent(), f, text)
        p.setPen(QPen(QColor("black"), 3))
        p.setBrush(QColor(col_str))
        p.drawPath(path)