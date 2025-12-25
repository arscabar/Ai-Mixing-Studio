import numpy as np
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any

class SourceType(Enum):
    ORIGINAL = 0
    AI_SEPARATED = 1
    IMPORTED = 2
    RESIDUAL = 3
    PROMPT_SEPARATED = 4  # [NEW] 프롬프트 분리 타입 추가

@dataclass
class Keyframe:
    time: float
    value: float
    interp: str = "linear"

@dataclass
class AutomationClip:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    param_type: str = "volume"
    start_time: float = 0.0
    end_time: float = 0.0
    start_value: float = 1.0
    end_value: float = 1.0
    description: str = ""
    keyframes: List[Keyframe] = field(default_factory=list)

    def add_keyframe(self, time: float, value: float):
        self.keyframes = [k for k in self.keyframes if abs(k.time - time) > 1e-4]
        self.keyframes.append(Keyframe(time, value))
        self.keyframes.sort(key=lambda k: k.time)

@dataclass
class TextEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = "New Text"
    start_time: float = 0.0
    end_time: float = 5.0
    x: int = 50
    y: int = 100
    font_family: str = "Arial"
    font_size: int = 40
    color_hex: str = "#FFFFFF"
    bold: bool = False
    italic: bool = False
    fade_in: float = 0.5
    fade_out: float = 0.5
    mask_mode: bool = False
    is_persistent: bool = False

@dataclass
class VisualEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = 0.0
    transition_duration: float = 1.0
    settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BackgroundEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = 0.0
    type: str = "image"
    path: str = ""
    fade_duration: float = 0.5

@dataclass
class SubtitleItem:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = 0.0
    end_time: float = 0.0
    text: str = ""
    language: str = ""
    font_size: int = 36
    color_hex: str = "#FFFF00"

class Track:
    def __init__(self, name: str, data: np.ndarray, sr: int = 44100, source_type: SourceType = SourceType.AI_SEPARATED):
        self.id = str(uuid.uuid4())
        self.name = name

        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        elif data.ndim == 2 and data.shape[0] == 2 and data.shape[1] > 2:
            data = data.T

        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"Track data must be stereo (N,2). got shape={getattr(data,'shape',None)}")

        self.data = data.astype(np.float32, copy=False)
        self.sr = int(sr)
        self.source_type = source_type

        self.volume = 1.0
        self.angle_deg = 0.0
        self.mute = False
        self.solo = False
        self.eq_low = 1.0
        self.eq_mid = 1.0
        self.eq_high = 1.0

        self.automations: List[AutomationClip] = []

        self.vis_min, self.vis_max = self._build_vis_cache(bucket=500)

    def update_data(self, new_data: np.ndarray):
        self.data = new_data.astype(np.float32, copy=False)
        self.vis_min, self.vis_max = self._build_vis_cache(bucket=500)

    def _build_vis_cache(self, bucket: int = 500):
        n = len(self.data)
        if n <= 0: return np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32)
        mono = self.data.mean(axis=1)
        if n < bucket:
            v = np.pad(mono, (0, max(0, 100 - n)), mode="constant")[:100].astype(np.float32, copy=False)
            return v, v
        m = n // bucket
        trimmed = mono[:m * bucket].reshape(m, bucket)
        return trimmed.min(axis=1).astype(np.float32, copy=False), trimmed.max(axis=1).astype(np.float32, copy=False)

    @property
    def duration(self) -> float:
        return len(self.data) / float(self.sr)

    def get_automated_value(self, param: str, current_time: float, default_val: float) -> float:
        for clip in self.automations:
            if clip.param_type == param and clip.keyframes:
                min_t = clip.keyframes[0].time
                max_t = clip.keyframes[-1].time
                if min_t <= current_time <= max_t:
                    return self._interpolate_keyframes(clip.keyframes, current_time)
        return default_val

    def _interpolate_keyframes(self, keys: List[Keyframe], t: float) -> float:
        for i in range(len(keys) - 1):
            k1 = keys[i]
            k2 = keys[i+1]
            if k1.time <= t <= k2.time:
                dur = k2.time - k1.time
                if dur <= 1e-6: return k1.value
                ratio = (t - k1.time) / dur
                return k1.value + (k2.value - k1.value) * ratio
        return keys[-1].value

class ProjectContext:
    def __init__(self):
        self.tracks: list[Track] = []
        self.sample_rate: int = 44100
        self.total_frames: int = 0
        self.is_playing: bool = False
        self.current_frame: int = 0
        self.master_volume: float = 1.0
        self.master_angle_deg: float = 0.0
        self.is_looping: bool = False
        self.loop_start: int = 0
        self.loop_end: int = 0
        
        self.text_events: List[TextEvent] = []
        self.visual_events: List[VisualEvent] = []
        self.bg_events: List[BackgroundEvent] = []
        self.subtitles: List[SubtitleItem] = []
        self.global_automations: List[AutomationClip] = []

    def add_track(self, track: Track):
        if not self.tracks: self.sample_rate = track.sr
        if track.sr != self.sample_rate: pass
        self.tracks.append(track)
        self.update_total_frames()

    def remove_track(self, track_id: str):
        self.tracks = [t for t in self.tracks if t.id != track_id]
        self.update_total_frames()
        
    def update_total_frames(self):
        if not self.tracks: self.total_frames = 0
        else: self.total_frames = max(len(t.data) for t in self.tracks)

    def clear(self):
        self.tracks = []
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.master_volume = 1.0
        self.master_angle_deg = 0.0
        self.is_looping = False
        self.loop_start = 0
        self.loop_end = 0
        self.text_events = []
        self.visual_events = []
        self.bg_events = []
        self.subtitles = []
        self.global_automations = []

    def get_global_value(self, param: str, time: float, default: float) -> float:
        for clip in self.global_automations:
            if clip.param_type == param and clip.keyframes:
                if clip.keyframes[0].time <= time <= clip.keyframes[-1].time:
                    for i in range(len(clip.keyframes)-1):
                        k1 = clip.keyframes[i]; k2 = clip.keyframes[i+1]
                        if k1.time <= time <= k2.time:
                            dur = k2.time - k1.time
                            if dur < 1e-6: return k1.value
                            r = (time - k1.time)/dur
                            return k1.value + (k2.value - k1.value)*r
        return default