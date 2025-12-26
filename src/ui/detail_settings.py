from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSlider, QSpinBox, 
                             QPushButton, QColorDialog, QComboBox, QCheckBox, 
                             QGroupBox, QHBoxLayout, QFormLayout, QDoubleSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal

class SpectrumSettingsWidget(QWidget):
    """
    [최종] 스펙트럼의 모든 속성(12종)을 제어하는 통합 패널
    """
    settings_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 1. Geometry (모양 및 크기)
        gb_geo = QGroupBox("1. Geometry & Layout")
        form_geo = QFormLayout()

        # Shape
        self.combo_shape = QComboBox()
        self.combo_shape.addItems(["Linear (Bar)", "Circle (Radial)"])
        self.combo_shape.currentIndexChanged.connect(self.emit_changes)
        form_geo.addRow("Mode:", self.combo_shape)

        # Density (Bar Count)
        self.spin_bars = QSpinBox()
        self.spin_bars.setRange(16, 1024); self.spin_bars.setValue(64); self.spin_bars.setSingleStep(16)
        self.spin_bars.valueChanged.connect(self.emit_changes)
        form_geo.addRow("Density (Bars):", self.spin_bars)

        # Masking Radius (Circle Only)
        self.slider_radius = QSlider(Qt.Orientation.Horizontal)
        self.slider_radius.setRange(0, 400); self.slider_radius.setValue(50)
        self.slider_radius.valueChanged.connect(self.emit_changes)
        form_geo.addRow("Inner Radius:", self.slider_radius)

        # Bar Gap (Spacing)
        self.slider_gap = QSlider(Qt.Orientation.Horizontal)
        self.slider_gap.setRange(0, 50); self.slider_gap.setValue(2)
        self.slider_gap.valueChanged.connect(self.emit_changes)
        form_geo.addRow("Bar Gap:", self.slider_gap)

        gb_geo.setLayout(form_geo)
        layout.addWidget(gb_geo)

        # 2. Physics & Frequency (반응성)
        gb_phys = QGroupBox("2. Physics & Frequency")
        form_phys = QFormLayout()

        # Sensitivity
        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setRange(10, 1000); self.slider_sens.setValue(150)
        self.slider_sens.valueChanged.connect(self.emit_changes)
        form_phys.addRow("Sensitivity:", self.slider_sens)

        # Smoothing (Decay)
        self.slider_smooth = QSlider(Qt.Orientation.Horizontal)
        self.slider_smooth.setRange(0, 95); self.slider_smooth.setValue(50) # 0.0 ~ 0.95
        self.slider_smooth.setToolTip("높을수록 천천히 내려옴")
        self.slider_smooth.valueChanged.connect(self.emit_changes)
        form_phys.addRow("Smoothing:", self.slider_smooth)

        # Log Scale
        self.chk_log = QCheckBox("Logarithmic Scale (Audio Friendly)")
        self.chk_log.setChecked(True)
        self.chk_log.toggled.connect(self.emit_changes)
        form_phys.addRow("Scale:", self.chk_log)

        # Frequency Range
        hbox_freq = QHBoxLayout()
        self.spin_min = QSpinBox(); self.spin_min.setRange(0, 500); self.spin_min.setValue(20)
        self.spin_max = QSpinBox(); self.spin_max.setRange(500, 24000); self.spin_max.setValue(16000)
        self.spin_min.valueChanged.connect(self.emit_changes)
        self.spin_max.valueChanged.connect(self.emit_changes)
        hbox_freq.addWidget(self.spin_min)
        hbox_freq.addWidget(QLabel("~"))
        hbox_freq.addWidget(self.spin_max)
        hbox_freq.addWidget(QLabel("Hz"))
        form_phys.addRow("Range:", hbox_freq)

        gb_phys.setLayout(form_phys)
        layout.addWidget(gb_phys)

        # 3. Appearance (스타일)
        gb_style = QGroupBox("3. Appearance")
        form_style = QFormLayout()
        
        # Color
        self.btn_color = QPushButton("Pick Color")
        self.color_val = "#00FF00"
        self.btn_color.setStyleSheet(f"background-color: {self.color_val}; color: black; font-weight: bold;")
        self.btn_color.clicked.connect(self.pick_color)
        form_style.addRow("Color:", self.btn_color)

        # Fill & Round
        hbox_style = QHBoxLayout()
        self.chk_fill = QCheckBox("Fill")
        self.chk_fill.setChecked(True)
        self.chk_fill.toggled.connect(self.emit_changes)
        
        self.chk_round = QCheckBox("Round Caps")
        self.chk_round.setChecked(False)
        self.chk_round.toggled.connect(self.emit_changes)
        
        hbox_style.addWidget(self.chk_fill)
        hbox_style.addWidget(self.chk_round)
        form_style.addRow("Style:", hbox_style)

        # Stroke Width
        self.spin_stroke = QSpinBox()
        self.spin_stroke.setRange(1, 20); self.spin_stroke.setValue(2)
        self.spin_stroke.valueChanged.connect(self.emit_changes)
        form_style.addRow("Stroke Width:", self.spin_stroke)

        gb_style.setLayout(form_style)
        layout.addWidget(gb_style)

        layout.addStretch()

    def load_settings(self, data: dict):
        self.blockSignals(True)
        idx = 1 if data.get('shape') == 'circle' else 0
        self.combo_shape.setCurrentIndex(idx)
        self.spin_bars.setValue(data.get('bar_count', 64))
        self.slider_radius.setValue(data.get('inner_radius', 50))
        self.slider_gap.setValue(data.get('bar_gap', 2))
        
        self.slider_sens.setValue(data.get('sensitivity', 150))
        self.slider_smooth.setValue(int(data.get('smoothing', 0.5) * 100))
        self.chk_log.setChecked(data.get('log_scale', True))
        self.spin_min.setValue(data.get('min_freq', 20))
        self.spin_max.setValue(data.get('max_freq', 16000))
        
        self.color_val = data.get('color', '#00FF00')
        self.btn_color.setStyleSheet(f"background-color: {self.color_val}; color: black;")
        self.chk_fill.setChecked(data.get('fill', True))
        self.chk_round.setChecked(data.get('round_caps', False))
        self.spin_stroke.setValue(data.get('stroke_width', 2))
        self.blockSignals(False)

    def pick_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            self.color_val = c.name()
            self.btn_color.setStyleSheet(f"background-color: {self.color_val}; color: black;")
            self.emit_changes()

    def emit_changes(self):
        shape_str = "circle" if self.combo_shape.currentIndex() == 1 else "linear"
        data = {
            "type": "spectrum",
            "shape": shape_str,
            "bar_count": self.spin_bars.value(),
            "inner_radius": self.slider_radius.value(),
            "bar_gap": self.slider_gap.value(),
            
            "sensitivity": self.slider_sens.value(),
            "smoothing": self.slider_smooth.value() / 100.0,
            "log_scale": self.chk_log.isChecked(),
            "min_freq": self.spin_min.value(),
            "max_freq": self.spin_max.value(),
            
            "color": self.color_val,
            "fill": self.chk_fill.isChecked(),
            "round_caps": self.chk_round.isChecked(),
            "stroke_width": self.spin_stroke.value()
        }
        self.settings_changed.emit(data)

# 텍스트 설정 클래스는 변경 없음 (기존 사용)
from PyQt6.QtWidgets import QLineEdit, QFontComboBox
class TextSettingsWidget(QWidget):
    settings_changed = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        gb = QGroupBox("Subtitle Settings")
        form = QFormLayout()
        
        self.txt_content = QLineEdit()
        self.txt_content.textChanged.connect(self.emit_changes)
        form.addRow("Text:", self.txt_content)
        
        self.font_combo = QFontComboBox()
        self.font_combo.currentFontChanged.connect(self.emit_changes)
        form.addRow("Font:", self.font_combo)
        
        self.spin_size = QSpinBox(); self.spin_size.setRange(10,300); self.spin_size.setValue(36)
        self.spin_size.valueChanged.connect(self.emit_changes)
        form.addRow("Size:", self.spin_size)
        
        self.btn_color = QPushButton("Color")
        self.col_val = "#FFFFFF"
        self.btn_color.clicked.connect(self.pick_color)
        form.addRow("Color:", self.btn_color)
        
        gb.setLayout(form)
        layout.addWidget(gb); layout.addStretch()

    def load_settings(self, data):
        self.blockSignals(True)
        self.txt_content.setText(data.get('text', ''))
        self.spin_size.setValue(data.get('font_size', 36))
        self.col_val = data.get('color', '#FFFFFF')
        self.blockSignals(False)

    def pick_color(self):
        c = QColorDialog.getColor()
        if c.isValid(): self.col_val = c.name(); self.emit_changes()

    def emit_changes(self):
        self.settings_changed.emit({
            'text': self.txt_content.text(),
            'font_family': self.font_combo.currentFont().family(),
            'font_size': self.spin_size.value(),
            'color': self.col_val
        })