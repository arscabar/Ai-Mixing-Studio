# src/models.py
import numpy as np
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any
from .core.automation import InterpolationType, calculate_value_at_time

class SourceType(Enum):
    ORIGINAL = 0; AI_SEPARATED = 1; IMPORTED = 2; RESIDUAL = 3

@dataclass
class Keyframe:
    time: float
    value: float
    interp: InterpolationType = InterpolationType.LINEAR

@dataclass
class AutomationClip:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    param_type: str = "volume"  # 'volume', 'pan', 'spectrum_scale'
    keyframes: List[Keyframe] = field(default_factory=list)
    color: str = "#FF0000"
    visible: bool = True
    
    def add_keyframe(self, time: float, value: float):
        self.keyframes = [k for k in self.keyframes if abs(k.time - time) > 0.001]
        self.keyframes.append(Keyframe(time, value))
        self.keyframes.sort(key=lambda k: k.time)

@dataclass
class TextEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    start_time: float = 0.0
    end_time: float = 5.0
    is_persistent: bool = False
    x: int = 50; y: int = 100
    # [설정 복구]
    font_family: str = "Arial"
    font_size: int = 60
    color_hex: str = "#FFFFFF"
    bold: bool = False
    italic: bool = False
    mask_mode: bool = False 

@dataclass
class VisualEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = 0.0
    transition_duration: float = 1.0
    settings: Dict[str, Any] = field(default_factory=dict)

class Track:
    def __init__(self, name: str, data: np.ndarray, sr: int = 44100, source_type: SourceType = SourceType.AI_SEPARATED):
        self.id = str(uuid.uuid4())
        self.name = name
        if data.ndim == 1: data = np.stack([data, data], axis=1)
        elif data.ndim > 2: data = data[:, :2]
        if data.shape[1] != 2: data = data.reshape(-1, 2) if data.size % 2 == 0 else np.zeros((100, 2))
        self.data = data.astype(np.float32, copy=False)
        self.sr = int(sr)
        self.source_type = source_type
        self.volume = 1.0; self.angle_deg = 0.0; self.mute = False; self.solo = False
        self.eq_low = 1.0; self.eq_mid = 1.0; self.eq_high = 1.0
        self.automations: List[AutomationClip] = []
        self.vis_min, self.vis_max = self._build_vis_cache()

    def update_data(self, new_data: np.ndarray):
        self.data = new_data.astype(np.float32, copy=False)
        self.vis_min, self.vis_max = self._build_vis_cache()

    def _build_vis_cache(self, bucket: int = 500):
        n = len(self.data)
        if n <= 0: return np.zeros(100), np.zeros(100)
        mono = self.data.mean(axis=1)
        m = n // bucket
        if m == 0: return mono, mono
        trimmed = mono[:m * bucket].reshape(m, bucket)
        return trimmed.min(axis=1), trimmed.max(axis=1)

    @property
    def duration(self) -> float:
        return len(self.data) / float(self.sr)

    def get_automated_value(self, param: str, current_time: float, default_val: float) -> float:
        target_clip = next((c for c in self.automations if c.param_type == param), None)
        if target_clip and target_clip.keyframes:
            from .core.automation import calculate_value_at_time
            return calculate_value_at_time(target_clip.keyframes, current_time, default_val)
        return default_val

class ProjectContext:
    def __init__(self):
        self.tracks: list[Track] = []
        self.sample_rate: int = 44100
        self.total_frames: int = 0
        self.is_playing: bool = False
        self.current_frame: int = 0
        self.master_volume: float = 1.0
        self.is_looping: bool = False
        self.loop_start: int = 0; self.loop_end: int = 0
        self.global_automations: List[AutomationClip] = []
        self.text_events: List[TextEvent] = []
        self.visual_events: List[VisualEvent] = []
        self.bg_events: List[Any] = []

    def add_track(self, track: Track):
        if not self.tracks: self.sample_rate = track.sr
        self.tracks.append(track)
        self.update_total_frames()

    def update_total_frames(self):
        if not self.tracks: self.total_frames = 0
        else: self.total_frames = max(len(t.data) for t in self.tracks)

    def clear(self):
        self.tracks = []; self.total_frames = 0; self.current_frame = 0
        self.is_playing = False; self.text_events = []; self.global_automations = []

    def get_global_value(self, param: str, current_time: float, default_val: float) -> float:
        target_clip = next((c for c in self.global_automations if c.param_type == param), None)
        if target_clip and target_clip.keyframes:
            from .core.automation import calculate_value_at_time
            return calculate_value_at_time(target_clip.keyframes, current_time, default_val)
        return default_val