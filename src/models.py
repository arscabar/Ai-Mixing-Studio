# src/models.py
import numpy as np
import uuid
from enum import Enum

class SourceType(Enum):
    ORIGINAL = 0
    AI_SEPARATED = 1
    IMPORTED = 2
    RESIDUAL = 3

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

        self._last_level = 0.0
        self._prev_sample = np.zeros((2,), dtype=np.float32)
        self.vis_min, self.vis_max = self._build_vis_cache(bucket=500)

    def update_data(self, new_data: np.ndarray):
        self.data = new_data.astype(np.float32, copy=False)
        self.vis_min, self.vis_max = self._build_vis_cache(bucket=500)

    def _build_vis_cache(self, bucket: int = 500):
        n = len(self.data)
        if n <= 0:
            z = np.zeros(100, dtype=np.float32)
            return z, z

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

    def add_track(self, track: Track):
        if not self.tracks:
            self.sample_rate = track.sr

        if track.sr != self.sample_rate:
            raise ValueError(f"Sample rate mismatch. project={self.sample_rate}, track={track.sr}")

        self.tracks.append(track)
        self.update_total_frames()

    def remove_track(self, track_id: str):
        self.tracks = [t for t in self.tracks if t.id != track_id]
        self.update_total_frames()
        
    def update_total_frames(self):
        if not self.tracks:
            self.total_frames = 0
        else:
            self.total_frames = max(len(t.data) for t in self.tracks)

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