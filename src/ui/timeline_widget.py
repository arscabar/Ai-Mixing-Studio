from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsLineItem, 
                             QGraphicsPathItem, QGraphicsTextItem, QWidget, QVBoxLayout, QMenu,
                             QGraphicsItem, QInputDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush

# --- 이벤트 박스 아이템 ---
class AutomationBoxItem(QGraphicsRectItem):
    def __init__(self, start_time, end_time, value, p_type, max_val, lane_height, y_offset, clip=None, sub_item=None, ctx=None, parent=None):
        """
        lane_height: 이 박스가 그려질 레인의 높이
        y_offset: 트랙 내에서 이 레인의 시작 Y 좌표
        """
        self.pps = 20
        self.start_time = start_time
        self.end_time = end_time
        self.value = value
        self.p_type = p_type
        self.max_val = max_val
        self.clip = clip
        self.sub_item = sub_item
        self.ctx = ctx
        self.lane_height = lane_height
        self.y_offset = y_offset
        
        x = start_time * self.pps
        w = max(5, (end_time - start_time) * self.pps)
        
        # 스타일 결정
        if p_type == 'subtitle':
            h_box = 30
            val_disp = sub_item.text if sub_item else "Sub"
            color = QColor(200, 200, 50, 150)
            base_y = y_offset + (lane_height - h_box)
        elif 'spec' in p_type: 
            h_box = 30
            val_disp = f"{p_type}"
            color = QColor(255, 100, 100, 150)
            base_y = y_offset + (lane_height - h_box)
        else: # Volume / Pan
            # 값에 따라 높이 비율 결정 (0.0 ~ 1.0)
            norm = max(0.0, min(1.0, value / max_val if max_val > 0 else 0))
            h_box = max(10, norm * (lane_height - 10)) # 최소 높이 10px 보장
            val_disp = f"{value:.1f}"
            
            color_map = {
                'volume': QColor(255, 200, 0, 160), 
                'pan': QColor(0, 255, 255, 160)
            }
            color = color_map.get(p_type, QColor(150, 150, 150, 150))
            
            # 바닥에서 솟아오르는 형태
            base_y = y_offset + (lane_height - h_box)

        super().__init__(0, 0, w, h_box, parent)
        self.setPos(x, base_y)
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor(255, 255, 255, 200), 1))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{p_type}: {val_disp}")
        self.drag_edge = None

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        if self.scene() and hasattr(self.scene(), 'parent_view'):
            self.scene().parent_view.event_selected.emit(self)

    def hoverMoveEvent(self, e):
        x = e.pos().x()
        w = self.rect().width()
        if x < 10: 
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            self.drag_edge = 'left'
        elif x > w - 10: 
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            self.drag_edge = 'right'
        else: 
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            self.drag_edge = None
        super().hoverMoveEvent(e)

    def mouseMoveEvent(self, e):
        # 드래그 중에는 데이터(update_model)를 건드리지 않고 UI만 변경 (렉 방지)
        if self.drag_edge:
            new_pos = self.mapToParent(e.pos())
            if self.drag_edge == 'left':
                new_x = min(new_pos.x(), (self.end_time - 0.1) * self.pps)
                right_x = self.pos().x() + self.rect().width()
                self.setPos(new_x, self.y())
                self.setRect(0, 0, right_x - new_x, self.rect().height())
            elif self.drag_edge == 'right':
                new_w = max(10, e.pos().x())
                self.setRect(0, 0, new_w, self.rect().height())
        else:
            # Y축(상하) 이동 잠금 -> 좌우로만 이동 가능
            orig_y = self.pos().y()
            super().mouseMoveEvent(e)
            self.setY(orig_y) 

    def mouseReleaseEvent(self, e):
        self.drag_edge = None
        super().mouseReleaseEvent(e)
        self.update_model()
        if self.scene() and hasattr(self.scene(), 'parent_view'):
            self.scene().parent_view.refresh_needed.emit()

    def update_model(self):
        new_s = max(0, self.pos().x() / self.pps)
        new_e = new_s + (self.rect().width() / self.pps)
        
        if self.sub_item:
            self.sub_item.start_time = new_s
            self.sub_item.end_time = new_e
            self.start_time = new_s
            self.end_time = new_e
            
        elif self.clip:
            old_s, old_e = self.start_time, self.end_time
            margin = 0.05
            
            # 1. 꼬리표(복귀 키프레임) 찾기
            trailing_key = None
            for k in self.clip.keyframes:
                if abs(k.time - (old_e + 0.01)) < margin:
                    trailing_key = k
                    break
            
            # 2. 기존 구간 삭제
            self.clip.keyframes = [
                k for k in self.clip.keyframes 
                if not (old_s - margin < k.time < old_e + margin + 0.02)
            ]
            
            # 3. 새 위치 생성
            self.clip.add_keyframe(new_s, self.value)
            self.clip.add_keyframe(new_e, self.value)
            
            # 4. 꼬리표 복구
            if trailing_key:
                self.clip.add_keyframe(new_e + 0.01, trailing_key.value)
            
            self.start_time = new_s
            self.end_time = new_e

    def delete_from_model(self):
        if self.sub_item:
            if self.sub_item in self.ctx.subtitles:
                self.ctx.subtitles.remove(self.sub_item)
        elif self.clip:
            s, e = self.start_time, self.end_time
            self.clip.keyframes = [k for k in self.clip.keyframes if k.time < s-0.05 or k.time > e+0.1]


# --- 마스터 트랙 (Subs 영역 포함) ---
class MasterTrackItem(QGraphicsRectItem):
    def __init__(self, ctx, width):
        super().__init__(0, 0, width, 150) # 높이 150
        self.ctx = ctx
        self.setPos(0, 30)
        self.setBrush(QBrush(QColor(30, 30, 40)))
        self.setPen(QPen(QColor(60, 60, 70)))
        
        t = QGraphicsTextItem("MASTER / SUBS", self)
        t.setDefaultTextColor(QColor(200, 200, 255))
        t.setPos(5, 5)
        
        # 구분선
        self.line = QGraphicsLineItem(0, 75, width, 75, self)
        self.line.setPen(QPen(QColor(100, 100, 100), 1, Qt.PenStyle.DashLine))
        
        self.drag_start = None
        self.refresh_visuals()

    def refresh_visuals(self):
        for c in self.childItems(): 
            if isinstance(c, AutomationBoxItem): self.scene().removeItem(c)
            
        # 1. Global (Spectrum 등) - 상단
        for p_type in ['spec_visible', 'spec_scale', 'spec_shape']:
            clip = next((c for c in self.ctx.global_automations if c.param_type == p_type), None)
            if clip:
                keys = sorted(clip.keyframes, key=lambda k: k.time)
                for i in range(0, len(keys), 2): 
                    if i+1 < len(keys):
                        k1, k2 = keys[i], keys[i+1]
                        if k2.time > k1.time:
                            AutomationBoxItem(k1.time, k2.time, k1.value, p_type, 1, 70, 0, clip, None, self.ctx, self)

        # 2. Subtitles - 하단
        for sub in self.ctx.subtitles:
            item = AutomationBoxItem(sub.start_time, sub.end_time, 0, 'subtitle', 1, 70, 75, None, sub, self.ctx, self)

    def mousePressEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier: 
            self.drag_start = e.pos().x()
        else: 
            super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self.drag_start is not None:
            end_x = e.pos().x()
            s, e_x = min(self.drag_start, end_x), max(self.drag_start, end_x)
            self.drag_start = None
            if e_x - s < 5: return
            
            menu = QMenu()
            act_spec = menu.addAction("Create Spectrum Event")
            act_text = menu.addAction("Create Text Event")
            action = menu.exec(e.screenPos())
            
            p_type = None
            if action == act_spec: p_type = 'spec_visible'
            elif action == act_text: p_type = 'subtitle'
            
            if p_type:
                if p_type == 'subtitle':
                    new_sub = SubtitleItem(start_time=s/20.0, end_time=e_x/20.0, text="New Text")
                    self.ctx.subtitles.append(new_sub)
                else:
                    self._apply_global(p_type, s/20.0, e_x/20.0, 1.0)
                
                if self.scene(): self.scene().parent_view.refresh_needed.emit()
        else: super().mouseReleaseEvent(e)

    def _apply_global(self, p_type, s, e, val):
        from ..models import AutomationClip
        clip = next((c for c in self.ctx.global_automations if c.param_type == p_type), None)
        if not clip:
            clip = AutomationClip(param_type=p_type)
            self.ctx.global_automations.append(clip)
        clip.add_keyframe(s, val)
        clip.add_keyframe(e, val)


# --- 개별 트랙 아이템 (2단 분리 + 병합 로직 + 기본값 숨김) ---
class TrackItem(QGraphicsRectItem):
    def __init__(self, track, width, index):
        super().__init__(0, 0, width, 130) # 높이 130
        self.track = track
        self.setPos(0, 200 + (index * 140)) 
        self.setBrush(QBrush(QColor(45, 45, 45)))
        self.setPen(QPen(QColor(30, 30, 30)))
        
        t = QGraphicsTextItem(track.name, self)
        t.setDefaultTextColor(QColor(220, 220, 220))
        t.setPos(5, 5)

        # 구분선
        self.mid_line = QGraphicsLineItem(0, 65, width, 65, self)
        self.mid_line.setPen(QPen(QColor(80, 80, 80), 1, Qt.PenStyle.DotLine))

        tv = QGraphicsTextItem("Vol", self)
        tv.setScale(0.8); tv.setDefaultTextColor(QColor(255, 200, 0)); tv.setPos(5, 45)
        
        tp = QGraphicsTextItem("Pan", self)
        tp.setScale(0.8); tp.setDefaultTextColor(QColor(0, 255, 255)); tp.setPos(5, 110)

        self.drag_start = None
        self.refresh_visuals()

    def refresh_visuals(self):
        for c in self.childItems(): 
            if isinstance(c, AutomationBoxItem): self.scene().removeItem(c)
        
        # Volume (상단)
        self._draw_boxes('volume', 2.0, 65, 0)
        # Pan (하단)
        self._draw_boxes('pan', 360.0, 65, 65)

    def _draw_boxes(self, p_type, max_val, lane_h, y_off):
        """
        [FIXED] 
        1. 연속된 구간 병합 (Merge)
        2. '기본값(Base Value)'과 동일한 구간은 그리지 않음 (0초부터 생기는 문제 해결)
        """
        clip = next((c for c in self.track.automations if c.param_type == p_type), None)
        if not clip or not clip.keyframes: return
        
        keys = sorted(clip.keyframes, key=lambda k: k.time)
        if not keys: return

        # 트랙의 현재 기본값 (Volume=1.0, Pan=0 등)
        def_v = self.track.volume if p_type == 'volume' else self.track.angle_deg

        # 1. 구간(Segment) 후보 추출
        segments = []
        for i in range(len(keys) - 1):
            k1 = keys[i]
            k2 = keys[i+1]
            
            # 값이 같고 시간 차이가 있을 때만 박스 후보
            if abs(k1.value - k2.value) < 0.001 and (k2.time - k1.time) > 0.05:
                segments.append({'s': k1.time, 'e': k2.time, 'v': k1.value})

        if not segments: return

        # 2. 연속 구간 병합
        merged = []
        curr = segments[0]
        for i in range(1, len(segments)):
            next_seg = segments[i]
            # 이어지고 & 값이 같으면 병합
            if abs(curr['e'] - next_seg['s']) < 0.02 and abs(curr['v'] - next_seg['v']) < 0.001:
                curr['e'] = next_seg['e']
            else:
                merged.append(curr)
                curr = next_seg
        merged.append(curr)

        # 3. 그리기 (단, 기본값과 같은 구간은 생략 -> 0초부터 그려지는 문제 방지)
        for m in merged:
            if abs(m['v'] - def_v) < 0.001: 
                continue # 기본값(배경)이면 그리지 않음
            
            AutomationBoxItem(m['s'], m['e'], m['v'], p_type, max_val, lane_h, y_off, clip, None, None, self)

    def mousePressEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier: 
            self.drag_start = e.pos()
        else: 
            super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self.drag_start is not None:
            start_pos = self.drag_start
            end_pos = e.pos()
            s = min(start_pos.x(), end_pos.x())
            e_x = max(start_pos.x(), end_pos.x())
            y_click = start_pos.y()
            
            self.drag_start = None
            if e_x - s < 5: return
            
            # Y좌표로 Volume/Pan 자동 판별
            if y_click < 65:
                p_type = 'volume'; max_v = 2.0
            else:
                p_type = 'pan'; max_v = 360.0
            
            val, ok = QInputDialog.getDouble(None, f"Set {p_type.title()}", "Value:", 1.0, 0.0, max_v, 2)
            if ok:
                self._apply_box(p_type, s/20.0, e_x/20.0, val)
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
        
        # 기존 겹치는 구간 정리
        clip.keyframes = [k for k in clip.keyframes if k.time < s or k.time > e+0.1]
        
        # 박스 생성
        clip.add_keyframe(s, val)
        clip.add_keyframe(e, val)
        # 꼬리표 (원상복구)
        clip.add_keyframe(e+0.01, def_v)


# --- 타임라인 룰러 & 뷰 ---
class TimelineRuler(QGraphicsRectItem):
    def __init__(self, duration_sec=300, pps=20):
        super().__init__(0, 0, duration_sec * pps, 30)
        self.pps = pps
        self.setBrush(QBrush(QColor(40, 40, 40)))
    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        painter.setPen(QColor(150, 150, 150))
        steps = int(self.rect().width() / self.pps)
        for i in range(steps + 1):
            x = i * self.pps
            h = 10 if i % 5 == 0 else 5
            painter.drawLine(int(x), 30-h, int(x), 30)
            if i % 5 == 0: painter.drawText(int(x)+2, 20, f"{i}s")

class PlayHeadItem(QGraphicsLineItem):
    def __init__(self):
        super().__init__(0, 0, 0, 1000)
        self.setPen(QPen(QColor(255, 0, 0), 2))
        self.setZValue(100)

class TimelineView(QGraphicsView):
    seek_requested = pyqtSignal(float)
    refresh_needed = pyqtSignal()
    event_selected = pyqtSignal(object) 
    request_settings_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.scene.parent_view = self 
        self.setScene(self.scene)
        self.setBackgroundBrush(QBrush(QColor(25, 25, 25)))
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.pps = 20
        self.playhead = PlayHeadItem()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Delete:
            for item in self.scene.selectedItems():
                if isinstance(item, AutomationBoxItem):
                    item.delete_from_model()
                    self.scene.removeItem(item)
            self.refresh_needed.emit()
        else: super().keyPressEvent(e)

    def refresh(self, ctx):
        self.scene.clear()
        self.playhead = PlayHeadItem()
        
        dur = 300
        if ctx.tracks: dur = max(dur, max(t.duration for t in ctx.tracks))
        
        self.ruler = TimelineRuler(dur, self.pps)
        self.scene.addItem(self.ruler)
        self.scene.addItem(self.playhead)

        width = dur * 20
        # MasterTrackItem
        master = MasterTrackItem(ctx, width)
        self.scene.addItem(master)
        
        y_c = 200 
        for i, t in enumerate(ctx.tracks):
            item = TrackItem(t, width, i)
            self.scene.addItem(item)
            y_c += 140 # 간격 증가
        
        self.scene.setSceneRect(0, 0, width, y_c + 50)
        self.playhead.setLine(0, 0, 0, y_c + 50)

    def set_position(self, sec): 
        self.playhead.setX(sec * 20)
        
    def mousePressEvent(self, e):
        pt = self.mapToScene(e.pos())
        if pt.y() < 30 and e.button() == Qt.MouseButton.LeftButton:
            self.seek_requested.emit(max(0, pt.x() / 20))
        super().mousePressEvent(e)

class TimelinePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.view = TimelineView()
        layout.addWidget(self.view)
        
    def refresh(self, ctx): 
        self.view.refresh(ctx)
        
    def update_playhead(self, sec): 
        self.view.set_position(sec)