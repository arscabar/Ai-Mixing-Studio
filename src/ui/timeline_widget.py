from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsLineItem, 
                             QGraphicsPathItem, QGraphicsTextItem, QWidget, QVBoxLayout, QInputDialog, 
                             QComboBox, QDialog, QFormLayout, QDoubleSpinBox, QDialogButtonBox, 
                             QLineEdit, QGraphicsItem, QPushButton, QColorDialog, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush, QWheelEvent, QPainterPath, QCursor
from ..models import AutomationClip, SubtitleItem, TextEvent

class BoxCreateDialog(QDialog):
    def __init__(self, mode='audio', parent=None, is_subtitle=False):
        super().__init__(parent)
        self.mode = mode
        self.is_subtitle = is_subtitle
        self.setWindowTitle("Event Settings")
        l = QFormLayout(self)
        
        self.combo_target = QComboBox()
        
        if self.is_subtitle:
            self.combo_target.addItems(["Subtitle"])
            self.combo_target.setEnabled(False)
        elif self.mode == 'master':
            self.combo_target.addItems([
                "Spectrum Visible (On/Off)", 
                "Spectrum Scale (Size)", 
                "Spectrum Shape (0:Linear, 1:Circ, 2:Line)"
            ])
        else:
            self.combo_target.addItems(["Volume", "Pan"])
            
        l.addRow("Target:", self.combo_target)
        
        self.spin_val = QDoubleSpinBox(); self.spin_val.setRange(0.0, 360.0); self.spin_val.setValue(1.0)
        l.addRow("Value:", self.spin_val)
        
        self.txt_content = QLineEdit(); self.txt_content.setPlaceholderText("Subtitle Text...")
        self.spin_size = QSpinBox(); self.spin_size.setRange(10, 200); self.spin_size.setValue(36)
        self.btn_color = QPushButton("Color"); self.col_val = QColor("#FFFF00")
        self.btn_color.setStyleSheet(f"background: {self.col_val.name()}; color: black;")
        self.btn_color.clicked.connect(self._pick_color)
        
        if self.is_subtitle:
            l.addRow("Text:", self.txt_content)
            l.addRow("Size:", self.spin_size)
            l.addRow("Color:", self.btn_color)
            self.spin_val.setVisible(False)
        else:
            self.txt_content.setVisible(False); self.spin_size.setVisible(False); self.btn_color.setVisible(False)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        l.addWidget(btns)

    def _pick_color(self):
        c = QColorDialog.getColor(self.col_val, self)
        if c.isValid():
            self.col_val = c
            self.btn_color.setStyleSheet(f"background: {c.name()}; color: {'black' if c.lightness() > 128 else 'white'};")

    def get_data(self):
        idx = self.combo_target.currentIndex()
        if self.is_subtitle:
            return 'subtitle', {
                'text': self.txt_content.text(),
                'size': self.spin_size.value(),
                'color': self.col_val.name()
            }
        elif self.mode == 'master':
            targets = ['spec_visible', 'spec_scale', 'spec_shape']
            return targets[idx], self.spin_val.value()
        else:
            return ('volume' if idx == 0 else 'pan'), self.spin_val.value()

class AutomationBoxItem(QGraphicsRectItem):
    def __init__(self, start_time, end_time, value, p_type, max_val, height, clip=None, sub_item=None, ctx=None, parent=None):
        self.pps = 20; self.start_time = start_time; self.end_time = end_time
        self.value = value; self.p_type = p_type; self.max_val = max_val
        self.clip = clip; self.sub_item = sub_item; self.ctx = ctx
        self.full_height = height
        
        x = start_time * self.pps
        w = max(5, (end_time - start_time) * self.pps)
        
        if p_type == 'subtitle':
            h_box = 30
            val_disp = sub_item.text if sub_item else "Sub"
            color = QColor(200, 200, 50, 150)
        else:
            norm = max(0.0, min(1.0, value / max_val if max_val > 0 else 0))
            h_box = max(4, norm * height)
            val_disp = f"{value:.1f}"
            color_map = {
                'spec_visible': QColor(100,255,100,150), 
                'spec_scale': QColor(50,150,255,150), 
                'spec_shape': QColor(255,100,255,150),
                'volume': QColor(255,200,0,150), 
                'pan': QColor(0,255,255,150)
            }
            color = color_map.get(p_type, QColor(150,150,150,150))

        y = height - h_box
        super().__init__(0, 0, w, h_box, parent)
        self.setPos(x, y)
        self.setBrush(QBrush(color)); self.setPen(QPen(QColor(255,255,255,200), 1))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{p_type}: {val_disp}")
        self.drag_edge = None

    def mouseDoubleClickEvent(self, e):
        is_sub = (self.p_type == 'subtitle')
        dlg = BoxCreateDialog(mode='master', is_subtitle=is_sub)
        
        if is_sub and self.sub_item:
            dlg.txt_content.setText(self.sub_item.text)
            dlg.spin_size.setValue(self.sub_item.font_size)
            dlg.col_val = QColor(self.sub_item.color_hex)
            dlg.btn_color.setStyleSheet(f"background:{self.sub_item.color_hex}")
        else:
            dlg.spin_val.setValue(self.value)

        if dlg.exec():
            t, val = dlg.get_data()
            if is_sub:
                self.sub_item.text = val['text']
                self.sub_item.font_size = val['size']
                self.sub_item.color_hex = val['color']
                self.setToolTip(f"Sub: {val['text']}")
            else:
                self.value = val
                self.update_model()
                norm = self.value / self.max_val if self.max_val > 0 else 0
                h_box = max(4, norm * self.full_height)
                self.setRect(0, 0, self.rect().width(), h_box)
                self.setPos(self.pos().x(), self.full_height - h_box)
                self.setToolTip(f"{self.p_type}: {self.value}")

    def hoverMoveEvent(self, e):
        x = e.pos().x(); w = self.rect().width()
        if x < 10: self.setCursor(Qt.CursorShape.SizeHorCursor); self.drag_edge = 'left'
        elif x > w - 10: self.setCursor(Qt.CursorShape.SizeHorCursor); self.drag_edge = 'right'
        else: self.setCursor(Qt.CursorShape.SizeAllCursor); self.drag_edge = None
        super().hoverMoveEvent(e)

    def mousePressEvent(self, e):
        if self.drag_edge: self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        else: self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.drag_edge:
            new_pos = self.mapToParent(e.pos())
            if self.drag_edge == 'left':
                new_x = min(new_pos.x(), (self.end_time - 0.1) * self.pps)
                right_x = self.pos().x() + self.rect().width()
                self.setPos(new_x, self.y())
                self.setRect(0, 0, right_x - new_x, self.rect().height())
            elif self.drag_edge == 'right':
                curr_x = self.pos().x()
                new_w = max(10, e.pos().x())
                self.setRect(0, 0, new_w, self.rect().height())
        else:
            super().mouseMoveEvent(e)
        
        self.update_model()

    def mouseReleaseEvent(self, e):
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.drag_edge = None
        super().mouseReleaseEvent(e)
        self.update_model()

    def update_model(self):
        s = max(0, self.pos().x() / self.pps)
        e = s + (self.rect().width() / self.pps)
        self.start_time = s; self.end_time = e
        
        if self.sub_item:
            self.sub_item.start_time = s; self.sub_item.end_time = e
        elif self.clip:
            margin = 0.05
            self.clip.keyframes = [k for k in self.clip.keyframes if k.time < s - margin or k.time > e + margin]
            def_val = 0.0 if self.p_type == 'spec_visible' else (1.0 if self.p_type=='spec_scale' else 0.0)
            self.clip.add_keyframe(s, self.value)
            self.clip.add_keyframe(e, self.value)
            self.clip.add_keyframe(e+0.01, def_val)

    def delete_from_model(self):
        if self.sub_item:
            self.ctx.subtitles.remove(self.sub_item)
        elif self.clip:
            s, e = self.start_time, self.end_time
            self.clip.keyframes = [k for k in self.clip.keyframes if k.time < s-0.05 or k.time > e+0.1]

class MasterTrackItem(QGraphicsRectItem):
    def __init__(self, ctx, width):
        super().__init__(0, 0, width, 120)
        self.ctx = ctx
        self.setPos(0, 30)
        self.setBrush(QBrush(QColor(40, 40, 50)))
        self.setPen(QPen(QColor(60, 60, 70)))
        t = QGraphicsTextItem("MASTER / SUBS", self)
        t.setDefaultTextColor(QColor(200, 200, 255)); t.setPos(5, 5)
        self.drag_start = None
        self.refresh_visuals()

    def refresh_visuals(self):
        for c in self.childItems(): 
            if isinstance(c, AutomationBoxItem): self.scene().removeItem(c)
            
        for p_type in ['spec_visible', 'spec_scale', 'spec_shape']:
            clip = next((c for c in self.ctx.global_automations if c.param_type == p_type), None)
            if clip:
                keys = sorted(clip.keyframes, key=lambda k: k.time)
                for i in range(len(keys)-1):
                    k1, k2 = keys[i], keys[i+1]
                    if (k2.time - k1.time) > 0.1 and abs(k1.value - k2.value) < 0.001:
                        if p_type == 'spec_visible' and k1.value < 0.5: continue 
                        max_v = 1.0 if p_type == 'spec_visible' else (5.0 if p_type == 'spec_scale' else 2.0)
                        AutomationBoxItem(k1.time, k2.time, k1.value, p_type, max_v, 120, clip, None, self.ctx, self)

        for sub in self.ctx.subtitles:
            AutomationBoxItem(sub.start_time, sub.end_time, 0, 'subtitle', 1, 120, None, sub, self.ctx, self)

    def mousePressEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier: self.drag_start = e.pos().x()
        else: super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self.drag_start is not None:
            end_x = e.pos().x()
            s, e_x = min(self.drag_start, end_x), max(self.drag_start, end_x)
            self.drag_start = None
            if e_x - s < 5: return
            
            types = ["Subtitle", "Spectrum Visible", "Spectrum Scale", "Spectrum Shape"]
            item, ok = QInputDialog.getItem(None, "Create Event", "Type:", types, 0, False)
            if ok:
                is_sub = (item == "Subtitle")
                dlg = BoxCreateDialog(mode='master', is_subtitle=is_sub)
                if dlg.exec():
                    target, val = dlg.get_data()
                    if is_sub:
                        self.ctx.subtitles.append(SubtitleItem(
                            start_time=s/20.0, end_time=e_x/20.0, 
                            text=val['text'], font_size=val['size'], color_hex=val['color']))
                    else:
                        self._apply_global(target, s/20.0, e_x/20.0, val)
                    
                    if self.scene(): self.scene().parent_view.refresh_needed.emit()
        else: super().mouseReleaseEvent(e)

    def _apply_global(self, p_type, s, e, val):
        from ..models import AutomationClip
        clip = next((c for c in self.ctx.global_automations if c.param_type == p_type), None)
        if not clip:
            clip = AutomationClip(param_type=p_type)
            self.ctx.global_automations.append(clip)
            def_v = 0.0 if p_type == 'spec_visible' else (1.0 if p_type=='spec_scale' else 0.0)
            clip.add_keyframe(0.0, def_v)
        
        clip.keyframes = [k for k in clip.keyframes if k.time < s or k.time > e+0.1]
        def_v = 0.0 if p_type == 'spec_visible' else (1.0 if p_type=='spec_scale' else 0.0)
        clip.add_keyframe(s, val); clip.add_keyframe(e, val); clip.add_keyframe(e+0.01, def_v)

class TrackItem(QGraphicsRectItem):
    def __init__(self, track, width, index):
        super().__init__(0, 0, width, 60)
        self.track = track; self.setPos(0, 160 + (index * 70))
        self.setBrush(QBrush(QColor(50, 50, 50))); self.setPen(QPen(QColor(30, 30, 30)))
        t = QGraphicsTextItem(track.name, self); t.setDefaultTextColor(QColor(200, 200, 200)); t.setPos(5, 5)
        self.drag_start = None
        self.refresh_visuals()

    def refresh_visuals(self):
        for c in self.childItems(): 
            if isinstance(c, AutomationBoxItem): self.scene().removeItem(c)
        self._draw_boxes('volume', 2.0)
        self._draw_boxes('pan', 360.0)

    def _draw_boxes(self, p_type, max_val):
        clip = next((c for c in self.track.automations if c.param_type == p_type), None)
        if not clip: return
        keys = sorted(clip.keyframes, key=lambda k: k.time)
        for i in range(len(keys)-1):
            k1, k2 = keys[i], keys[i+1]
            if abs(k1.value - k2.value) < 0.001 and (k2.time - k1.time) > 0.1:
                AutomationBoxItem(k1.time, k2.time, k1.value, p_type, max_val, 60, clip, None, None, self)

    def mousePressEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier: self.drag_start = e.pos().x()
        else: super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self.drag_start is not None:
            end_x = e.pos().x()
            s, e_x = min(self.drag_start, end_x), max(self.drag_start, end_x)
            self.drag_start = None
            if e_x - s < 5: return
            
            dlg = BoxCreateDialog(mode='audio')
            if dlg.exec():
                target, val = dlg.get_data()
                self._apply_box(target, s/20.0, e_x/20.0, float(val))
                if self.scene(): self.scene().parent_view.refresh_needed.emit()
        else: super().mouseReleaseEvent(e)

    def _apply_box(self, p_type, s, e, val):
        from ..models import AutomationClip
        clip = next((c for c in self.track.automations if c.param_type == p_type), None)
        if not clip:
            clip = AutomationClip(param_type=p_type)
            self.track.automations.append(clip)
            def_v = self.track.volume if p_type=='volume' else self.track.angle_deg
            clip.add_keyframe(0.0, def_v)
        
        def_v = self.track.volume if p_type=='volume' else self.track.angle_deg
        clip.keyframes = [k for k in clip.keyframes if k.time < s or k.time > e+0.1]
        clip.add_keyframe(s, val); clip.add_keyframe(e, val); clip.add_keyframe(e+0.01, def_v)

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
    seek_requested = pyqtSignal(float); refresh_needed = pyqtSignal()
    request_settings_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self); self.scene.parent_view = self 
        self.setScene(self.scene); self.setBackgroundBrush(QBrush(QColor(25, 25, 25)))
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.pps = 20; self.playhead = PlayHeadItem()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Delete:
            for item in self.scene.selectedItems():
                if isinstance(item, AutomationBoxItem):
                    item.delete_from_model(); self.scene.removeItem(item)
            self.refresh_needed.emit()
        else: super().keyPressEvent(e)

    def refresh(self, ctx):
        self.scene.clear()
        
        # [FIX] Playhead 재성성 (clear()시 삭제되므로)
        self.playhead = PlayHeadItem()
        
        dur = 300
        if ctx.tracks: dur = max(dur, max(t.duration for t in ctx.tracks))
        
        # Ruler 생성 및 추가
        self.ruler = TimelineRuler(dur, self.pps)
        self.scene.addItem(self.ruler)
        self.scene.addItem(self.playhead)

        width = dur * 20
        master = MasterTrackItem(ctx, width)
        self.scene.addItem(master)
        
        y_c = 160
        for i, t in enumerate(ctx.tracks):
            item = TrackItem(t, width, i)
            self.scene.addItem(item)
            y_c += 70
        
        self.scene.setSceneRect(0, 0, width, y_c + 50)
        self.playhead.setLine(0, 0, 0, y_c + 50)

    def set_position(self, sec): self.playhead.setX(sec * 20)
    def mousePressEvent(self, e):
        pt = self.mapToScene(e.pos())
        if pt.y() < 30 and e.button() == Qt.MouseButton.LeftButton:
            self.seek_requested.emit(max(0, pt.x() / 20))
        super().mousePressEvent(e)

class TimelinePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        self.view = TimelineView(); layout.addWidget(self.view)
    def refresh(self, ctx): self.view.refresh(ctx)
    def update_playhead(self, sec): self.view.set_position(sec)