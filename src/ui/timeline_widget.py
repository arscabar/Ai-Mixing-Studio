# src/ui/timeline_widget.py
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsLineItem, 
                             QGraphicsPathItem, QGraphicsTextItem, QWidget, QVBoxLayout, QInputDialog, 
                             QComboBox, QDialog, QFormLayout, QDoubleSpinBox, QDialogButtonBox, 
                             QLineEdit, QGraphicsItem, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush, QWheelEvent, QPainterPath, QCursor
from ..models import AutomationClip, TextEvent

class BoxCreateDialog(QDialog):
    def __init__(self, mode='audio', parent=None):
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle("Set Value")
        l = QFormLayout(self)
        
        self.combo_target = QComboBox()
        if self.mode == 'master':
            self.combo_target.addItems(["Spectrum Scale", "Text Event"]) 
        else:
            self.combo_target.addItems(["Volume", "Pan (Direction)"])
            
        l.addRow("Target:", self.combo_target)
        
        self.spin_val = QDoubleSpinBox(); self.spin_val.setRange(0.0, 360.0); self.spin_val.setValue(1.0)
        self.txt_val = QLineEdit(); self.txt_val.setPlaceholderText("Enter Text...")
        
        l.addRow("Value/Text:", self.spin_val)
        l.addRow("", self.txt_val)
        
        if self.mode == 'master':
            self.btn_detail = QPushButton("Open Detailed Settings")
            self.btn_detail.clicked.connect(self.reject_and_open_settings)
            l.addRow("", self.btn_detail)
        
        self.open_settings = False
        self.combo_target.currentIndexChanged.connect(self._on_type_change)
        self._on_type_change(0)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        l.addWidget(btns)

    def _on_type_change(self, idx):
        if self.mode == 'master':
            is_text = (idx == 1)
            self.spin_val.setVisible(not is_text)
            self.txt_val.setVisible(is_text)
        else:
            self.spin_val.setVisible(True)
            self.txt_val.setVisible(False)

    def reject_and_open_settings(self):
        self.open_settings = True
        self.accept()

    def get_data(self):
        idx = self.combo_target.currentIndex()
        if self.mode == 'master':
            target = 'text' if idx == 1 else 'spectrum_scale'
            val = self.txt_val.text() if idx == 1 else self.spin_val.value()
            return target, val, self.open_settings
        else:
            target = 'volume' if idx == 0 else 'pan'
            return target, self.spin_val.value(), False

class AutomationBoxItem(QGraphicsRectItem):
    def __init__(self, start_time, end_time, value, p_type, max_val, height, clip, ctx=None, parent=None):
        self.pps = 20
        self.start_time = start_time
        self.end_time = end_time
        self.value = value
        self.p_type = p_type
        self.max_val = max_val
        self.full_height = height
        self.clip = clip
        self.ctx = ctx
        
        x = start_time * self.pps
        w = (end_time - start_time) * self.pps
        if w < 5: w = 5
        
        if isinstance(value, str):
            h_box = height * 0.8
            val_disp = value
            color = QColor(150, 100, 200, 150)
        else:
            norm = value / max_val if max_val > 0 else 0
            norm = max(0.0, min(1.0, norm))
            h_box = max(4, norm * height)
            val_disp = f"{value:.2f}"
            if p_type == 'spectrum_scale': color = QColor(100, 255, 100, 150)
            elif p_type == 'volume': color = QColor(255, 200, 0, 150)
            else: color = QColor(0, 255, 255, 150)

        y = height - h_box
        super().__init__(0, 0, w, h_box, parent)
        self.setPos(x, y)
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor(255,255,255,200), 1))
        
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | 
                      QGraphicsItem.GraphicsItemFlag.ItemIsMovable | 
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{p_type}: {val_disp}\n(Double-click to Edit)")
        
        self.drag_edge = None
        self.drag_limits = (0.0, 99999.0)

    def mouseDoubleClickEvent(self, e):
        if self.p_type == 'spectrum_scale':
            if self.scene() and hasattr(self.scene(), 'parent_view'):
                self.scene().parent_view.request_settings_signal.emit()
        elif isinstance(self.value, str):
            txt, ok = QInputDialog.getText(None, "Edit Text", "Content:", text=self.value)
            if ok:
                self.value = txt
                self.update_model()
                self.setToolTip(f"Text: {txt}")
        else:
            val, ok = QInputDialog.getDouble(None, "Edit Value", f"Set {self.p_type}:", value=self.value)
            if ok:
                self.value = val
                self.update_model()
                norm = self.value / self.max_val if self.max_val > 0 else 0
                h_box = max(4, norm * self.full_height)
                self.setRect(0, 0, self.rect().width(), h_box)
                self.setPos(self.pos().x(), self.full_height - h_box)

    def hoverMoveEvent(self, e):
        x = e.pos().x(); w = self.rect().width()
        if x < 10:
            self.setCursor(Qt.CursorShape.SizeHorCursor); self.drag_edge = 'left'
        elif x > w - 10:
            self.setCursor(Qt.CursorShape.SizeHorCursor); self.drag_edge = 'right'
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor); self.drag_edge = None
        super().hoverMoveEvent(e)

    def mousePressEvent(self, e):
        self._calc_drag_limits()
        if self.drag_edge: self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        else: self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        super().mousePressEvent(e)

    def _calc_drag_limits(self):
        min_t = 0.0
        max_t = 99999.0
        epsilon = 0.05

        if self.clip: 
            keys = sorted(self.clip.keyframes, key=lambda k: k.time)
            for k in keys:
                if k.time < self.start_time - epsilon: min_t = max(min_t, k.time)
                if k.time > self.end_time + epsilon: max_t = min(max_t, k.time)
        
        elif self.ctx and self.p_type == 'text':
            for evt in self.ctx.text_events:
                if evt.text == self.value and abs(evt.start_time - self.start_time) < 0.01: continue
                if evt.end_time < self.start_time - epsilon: min_t = max(min_t, evt.end_time)
                if evt.start_time > self.end_time + epsilon: max_t = min(max_t, evt.start_time)

        self.drag_limits = (min_t + 0.1, max_t - 0.1)

    def mouseMoveEvent(self, e):
        min_t, max_t = self.drag_limits
        
        if self.drag_edge:
            new_pos = self.mapToParent(e.pos())
            if self.drag_edge == 'left':
                new_x = max(min_t * self.pps, min(new_pos.x(), (self.end_time - 0.5) * self.pps))
                right_x = self.pos().x() + self.rect().width()
                if new_x < right_x - 10:
                    self.setPos(new_x, self.y())
                    self.setRect(0, 0, right_x - new_x, self.rect().height())
            elif self.drag_edge == 'right':
                curr_x = self.pos().x()
                max_w = (max_t * self.pps) - curr_x
                new_w = min(max_w, max(10, e.pos().x()))
                self.setRect(0, 0, new_w, self.rect().height())
            
            s = self.pos().x()/self.pps
            e_t = (self.pos().x()+self.rect().width())/self.pps
            self.setToolTip(f"{self.p_type}: {self.value}\n{s:.1f}s ~ {e_t:.1f}s")
        else:
            super().mouseMoveEvent(e)
            new_x = self.pos().x()
            new_s = new_x / self.pps
            new_e = new_s + (self.rect().width() / self.pps)
            
            if new_s < min_t: new_x = min_t * self.pps
            if new_e > max_t: new_x = (max_t * self.pps) - self.rect().width()
            self.setPos(new_x, self.y())

    def mouseReleaseEvent(self, e):
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.drag_edge = None
        super().mouseReleaseEvent(e)
        self.update_model()

    def _get_default_value(self):
        """[핵심] 박스가 끝난 후 돌아갈 기본값 가져오기 (사용자 설정값)"""
        parent = self.parentItem()
        # Audio Track인 경우 (TrackItem)
        if hasattr(parent, 'track'):
            if self.p_type == 'volume': return parent.track.volume
            if self.p_type == 'pan': return parent.track.angle_deg
        # Master Track인 경우 (Spectrum)
        return 1.0 # Spectrum default scale

    def update_model(self):
        new_start = max(0, self.pos().x() / self.pps)
        new_end = new_start + (self.rect().width() / self.pps)
        self.start_time = new_start
        self.end_time = new_end
        
        if self.p_type == 'text':
            for evt in self.ctx.text_events:
                if evt.text == self.value: 
                    evt.start_time = new_start
                    evt.end_time = new_end
            return

        margin = 0.05
        # 겹치는 구간 삭제
        self.clip.keyframes = [k for k in self.clip.keyframes if k.time < new_start - margin or k.time > new_end + margin]
        
        # [수정] 박스 끝난 뒤 돌아갈 값을 현재 트랙의 설정값으로 지정
        def_val = self._get_default_value()
        
        self.clip.add_keyframe(new_start, self.value)
        self.clip.add_keyframe(new_end, self.value)
        self.clip.add_keyframe(new_end + 0.01, def_val) # Return to user-set default

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            self.setPen(QPen(QColor(255, 0, 0), 2)) if value else self.setPen(QPen(QColor(255, 255, 255, 200), 1))
        return super().itemChange(change, value)

    def delete_from_model(self):
        if self.p_type == 'text':
            self.ctx.text_events = [e for e in self.ctx.text_events if not (e.text == self.value and abs(e.start_time - self.start_time) < 0.1)]
        elif self.clip:
            s, e = self.start_time, self.end_time
            self.clip.keyframes = [k for k in self.clip.keyframes if k.time < s-0.05 or k.time > e+0.1]

class MasterTrackItem(QGraphicsRectItem):
    def __init__(self, ctx, width):
        super().__init__(0, 0, width, 80)
        self.ctx = ctx
        self.setPos(0, 40)
        self.setBrush(QBrush(QColor(40, 40, 50)))
        self.setPen(QPen(QColor(60, 60, 70)))
        t = QGraphicsTextItem("GLOBAL (Spec / Text)", self)
        t.setDefaultTextColor(QColor(200, 200, 255)); t.setPos(5, 5)
        self.drag_start_x = None
        self.refresh_visuals()

    def refresh_visuals(self):
        for child in self.childItems():
            if isinstance(child, AutomationBoxItem): self.scene().removeItem(child)
            
        clip = next((c for c in self.ctx.global_automations if c.param_type == 'spectrum_scale'), None)
        if clip:
            keys = sorted(clip.keyframes, key=lambda k: k.time)
            i=0
            while i < len(keys)-1:
                k1, k2 = keys[i], keys[i+1]
                if (k2.time - k1.time) > 0.1 and abs(k1.value - k2.value) < 0.001:
                    AutomationBoxItem(k1.time, k2.time, k1.value, 'spectrum_scale', 5.0, 80, clip, self.ctx, self)
                i+=1

        for evt in self.ctx.text_events:
            end = evt.end_time if not evt.is_persistent else (self.rect().width()/20.0)
            b = AutomationBoxItem(evt.start_time, end, evt.text, 'text', 1.0, 80, None, self.ctx, self)
            b.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

    def mousePressEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier: self.drag_start_x = e.pos().x()
        else: super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self.drag_start_x is not None:
            end_x = e.pos().x()
            s, e_x = min(self.drag_start_x, end_x), max(self.drag_start_x, end_x)
            self.drag_start_x = None
            if e_x - s < 5: return
            
            dlg = BoxCreateDialog(mode='master')
            if dlg.exec():
                target, val, open_set = dlg.get_data()
                view = self.scene().parent_view if self.scene() else None
                
                if target == 'spectrum_scale':
                    self._apply_spec(s/20.0, e_x/20.0, float(val))
                elif target == 'text':
                    from ..models import TextEvent
                    self.ctx.text_events.append(TextEvent(text=str(val), start_time=s/20.0, end_time=e_x/20.0))
                
                if view:
                    if open_set: view.request_settings_signal.emit()
                    view.refresh_needed.emit()
        else: super().mouseReleaseEvent(e)

    def _apply_spec(self, s, e, val):
        from ..models import AutomationClip
        clip = next((c for c in self.ctx.global_automations if c.param_type == 'spectrum_scale'), None)
        if not clip:
            clip = AutomationClip(param_type='spectrum_scale')
            self.ctx.global_automations.append(clip)
            clip.add_keyframe(0.0, 1.0)
        
        clip.keyframes = [k for k in clip.keyframes if k.time < s or k.time > e+0.1]
        clip.add_keyframe(s, val); clip.add_keyframe(e, val); clip.add_keyframe(e+0.01, 1.0)

class TrackItem(QGraphicsRectItem):
    def __init__(self, track, width, index):
        super().__init__(0, 0, width, 60)
        self.track = track
        self.setPos(0, 130 + (index * 70))
        self.setBrush(QBrush(QColor(50, 50, 50)))
        self.setPen(QPen(QColor(30, 30, 30)))
        t = QGraphicsTextItem(track.name, self); t.setDefaultTextColor(QColor(200, 200, 200)); t.setPos(5, 5)
        self.drag_start_x = None
        self.refresh_visuals()

    def refresh_visuals(self):
        for child in self.childItems():
            if isinstance(child, AutomationBoxItem) or isinstance(child, QGraphicsPathItem): 
                self.scene().removeItem(child)
        self._draw_bg()
        self._draw_boxes('volume', 2.0)
        self._draw_boxes('pan', 360.0)

    def _draw_bg(self):
        self._draw_curve('volume', QColor(255, 200, 0, 30), 2.0)
        self._draw_curve('pan', QColor(0, 255, 255, 30), 360.0)

    def _draw_curve(self, p_type, color, max_val):
        clip = next((c for c in self.track.automations if c.param_type == p_type), None)
        if not clip or not clip.keyframes: return
        path = QPainterPath(); pps = 20; h = 60
        keys = sorted(clip.keyframes, key=lambda k: k.time)
        def get_y(val): return h - (max(0.0, min(1.0, val / max_val if max_val>0 else 0)) * h)
        path.moveTo(keys[0].time * pps, get_y(keys[0].value))
        for k in keys[1:]: path.lineTo(k.time * pps, get_y(k.value))
        item = QGraphicsPathItem(path, self); item.setPen(QPen(color, 1))

    def _draw_boxes(self, p_type, max_val):
        clip = next((c for c in self.track.automations if c.param_type == p_type), None)
        if not clip or not clip.keyframes: return
        keys = sorted(clip.keyframes, key=lambda k: k.time)
        i=0
        while i < len(keys)-1:
            k1, k2 = keys[i], keys[i+1]
            if abs(k1.value - k2.value) < 0.001 and (k2.time - k1.time) > 0.1:
                AutomationBoxItem(k1.time, k2.time, k1.value, p_type, max_val, 60, clip, None, self)
            i+=1

    def mousePressEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier: self.drag_start_x = e.pos().x()
        else: super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self.drag_start_x is not None:
            end_x = e.pos().x()
            s, e_x = min(self.drag_start_x, end_x), max(self.drag_start_x, end_x)
            self.drag_start_x = None
            if e_x - s < 5: return
            
            dlg = BoxCreateDialog(mode='audio')
            if dlg.exec():
                target, val, _ = dlg.get_data()
                view = self.scene().parent_view if self.scene() else None
                self._apply_box(target, s/20.0, e_x/20.0, float(val))
                if view: view.refresh_needed.emit()
        else: super().mouseReleaseEvent(e)

    def _apply_box(self, p_type, s, e, val):
        from ..models import AutomationClip
        clip = next((c for c in self.track.automations if c.param_type == p_type), None)
        if not clip:
            clip = AutomationClip(param_type=p_type)
            self.track.automations.append(clip)
            d = self.track.volume if p_type=='volume' else self.track.angle_deg
            clip.add_keyframe(0.0, d)
        
        # [수정] 박스 끝난 뒤 돌아갈 기본값 (사용자 설정값)
        d = self.track.volume if p_type=='volume' else self.track.angle_deg
        
        clip.keyframes = [k for k in clip.keyframes if k.time < s or k.time > e+0.1]
        clip.add_keyframe(s, val)
        clip.add_keyframe(e, val)
        clip.add_keyframe(e+0.01, d) # Return to user default

class TimelineRuler(QGraphicsRectItem):
    def __init__(self, duration_sec=300, pps=20):
        super().__init__(0, 0, duration_sec * pps, 30)
        self.pps = pps; self.setBrush(QBrush(QColor(40, 40, 40)))
    def paint(self, painter, option, widget):
        super().paint(painter, option, widget); painter.setPen(QColor(150, 150, 150))
        steps = int(self.rect().width() / self.pps)
        for i in range(steps + 1):
            x = i * self.pps; h = 10 if i % 5 == 0 else 5
            painter.drawLine(int(x), 30-h, int(x), 30)
            if i % 5 == 0: painter.drawText(int(x)+2, 20, f"{i}s")

class PlayHeadItem(QGraphicsLineItem):
    def __init__(self):
        super().__init__(0, 0, 0, 1000); self.setPen(QPen(QColor(255, 0, 0), 2)); self.setZValue(100)

class TimelineView(QGraphicsView):
    seek_requested = pyqtSignal(float)
    refresh_needed = pyqtSignal()
    request_settings_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self); self.scene.parent_view = self 
        self.setScene(self.scene); self.setBackgroundBrush(QBrush(QColor(25, 25, 25)))
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.pps = 20; self.duration = 300; self.ruler = TimelineRuler(self.duration, self.pps); self.playhead = PlayHeadItem()
        self._init_scene()

    def _init_scene(self):
        self.scene.clear(); self.scene.addItem(self.ruler); self.scene.addItem(self.playhead)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Delete:
            for item in self.scene.selectedItems():
                if isinstance(item, AutomationBoxItem):
                    item.delete_from_model()
                    self.scene.removeItem(item)
            self.refresh_needed.emit()
        else: super().keyPressEvent(e)

    def refresh(self, ctx):
        self.scene.clear(); self.ruler = TimelineRuler(self.duration, self.pps); self.playhead = PlayHeadItem()
        self.scene.addItem(self.ruler); self.scene.addItem(self.playhead)
        
        width = self.duration * self.pps
        master = MasterTrackItem(ctx, width)
        self.scene.addItem(master)
        
        y_cursor = 130
        if hasattr(ctx, 'tracks'):
            for i, t in enumerate(ctx.tracks):
                w = max(width, t.duration * self.pps)
                item = TrackItem(t, w, i)
                self.scene.addItem(item)
                y_cursor += 70
        
        self.scene.setSceneRect(0, 0, width, y_cursor + 50)
        self.playhead.setLine(0, 0, 0, y_cursor + 50)

    def set_position(self, sec): self.playhead.setX(sec * self.pps)
    def mousePressEvent(self, e):
        pt = self.mapToScene(e.pos())
        if pt.y() < 30 and e.button() == Qt.MouseButton.LeftButton:
            self.seek_requested.emit(max(0, pt.x() / self.pps))
        super().mousePressEvent(e)

class TimelinePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        self.view = TimelineView(); layout.addWidget(self.view)
    def refresh(self, ctx): self.view.refresh(ctx)
    def update_playhead(self, sec): self.view.set_position(sec)