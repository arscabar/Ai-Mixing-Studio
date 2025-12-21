# src/audio_engine.py
import math
import numpy as np
import sounddevice as sd
import soundfile as sf
from .models import ProjectContext

class AudioEngine:
    def __init__(self, context: ProjectContext):
        self.ctx = context
        self.stream: sd.OutputStream | None = None
        self.block_size = 2048
        self.dtype = "float32"
        self.vis_buffer: np.ndarray | None = None

    def start(self):
        if self.stream is None:
            self.stream = sd.OutputStream(
                samplerate=self.ctx.sample_rate,
                blocksize=self.block_size,
                channels=2,
                dtype=self.dtype,
                callback=self._audio_callback,
            )
        if not self.stream.active:
            self.stream.start()
        self.ctx.is_playing = True

    def stop(self):
        if self.stream and self.stream.active:
            self.stream.stop()
        self.ctx.is_playing = False

    def seek(self, frame_idx: int):
        self.ctx.current_frame = min(max(0, int(frame_idx)), int(self.ctx.total_frames))

    def _audio_callback(self, outdata, frames, time, status):
        if not self.ctx.is_playing or not self.ctx.tracks or self.ctx.total_frames <= 0:
            outdata.fill(0)
            self.vis_buffer = np.zeros((frames, 2), dtype=np.float32)
            return

        mix_buffer = self._mix_chunk(self.ctx.current_frame, frames)
        outdata[:] = mix_buffer
        self.vis_buffer = mix_buffer.copy()
        
        self.ctx.current_frame += frames

        if self.ctx.current_frame >= self.ctx.total_frames:
            if not self.ctx.is_looping:
                self.ctx.is_playing = False
                self.ctx.current_frame = 0
                raise sd.CallbackStop
            else:
                if self.ctx.loop_end > self.ctx.loop_start:
                     if self.ctx.current_frame >= self.ctx.loop_end:
                         self.ctx.current_frame = self.ctx.loop_start
                else:
                    self.ctx.current_frame = 0

    @staticmethod
    def _angle_pan_front(angle_deg: float):
        theta = math.radians(float(angle_deg) % 360.0)
        return float(math.sin(theta)), float(math.cos(theta))

    @staticmethod
    def _apply_back_cues(y: np.ndarray, backness: float, prev_sample: np.ndarray):
        if backness <= 1e-6:
            return y, y[-1].astype(np.float32, copy=True)
        b = float(np.clip(backness, 0.0, 1.0))
        k = 0.85 * b
        d = np.empty_like(y)
        d[0, :] = y[0, :] - prev_sample
        d[1:, :] = y[1:, :] - y[:-1, :]
        y = y - k * d
        cf = 0.35 * b
        mono = y.mean(axis=1, keepdims=True)
        y = (1.0 - cf) * y + cf * mono
        y *= (1.0 - 0.12 * b)
        return y, y[-1].astype(np.float32, copy=True)

    def _mix_chunk(self, start_frame: int, n_frames: int) -> np.ndarray:
        mix = np.zeros((n_frames, 2), dtype=np.float32)
        is_any_solo = any(t.solo for t in self.ctx.tracks)
        master_angle = float(getattr(self.ctx, "master_angle_deg", 0.0))

        for t in self.ctx.tracks:
            if t.mute: continue
            if is_any_solo and not t.solo: continue

            track_len = len(t.data)
            if start_frame >= track_len: continue

            end_frame = min(start_frame + n_frames, track_len)
            read_len = end_frame - start_frame
            chunk = t.data[start_frame:end_frame]
            if read_len < n_frames:
                chunk = np.pad(chunk, ((0, n_frames - read_len), (0, 0)), mode="constant")

            angle_deg = (float(getattr(t, "angle_deg", 0.0)) + master_angle) % 360.0
            pan, front = self._angle_pan_front(angle_deg)
            backness = (1.0 - front) * 0.5

            angle_lr = (pan + 1.0) * (math.pi / 4.0)
            left_gain = math.cos(angle_lr)
            right_gain = math.sin(angle_lr)

            vol = float(max(0.0, t.volume))
            y = chunk.astype(np.float32, copy=False).copy()
            y[:, 0] *= vol * left_gain
            y[:, 1] *= vol * right_gain

            prev = getattr(t, "_prev_sample", None)
            if prev is None or not isinstance(prev, np.ndarray) or prev.shape != (2,):
                prev = np.zeros((2,), dtype=np.float32)

            y, new_prev = self._apply_back_cues(y, backness, prev)
            t._prev_sample = new_prev

            try:
                t._last_level = float(np.max(np.abs(y)))
            except Exception:
                t._last_level = 0.0

            mix += y

        master_vol = float(max(0.0, getattr(self.ctx, "master_volume", 1.0)))
        mix *= master_vol
        return np.tanh(mix).astype(np.float32, copy=False)

    def export_mix_to_wav(self, file_path: str) -> bool:
        if not self.ctx.tracks or self.ctx.total_frames <= 0: return False
        try:
            current = 0
            chunk_size = 4096
            total = int(self.ctx.total_frames)
            with sf.SoundFile(file_path, mode="w", samplerate=int(self.ctx.sample_rate), channels=2, subtype="PCM_16") as f:
                while current < total:
                    read_frames = min(chunk_size, total - current)
                    audio_chunk = self._mix_chunk(current, read_frames)
                    f.write(audio_chunk)
                    current += read_frames
            return True
        except Exception as e:
            print(f"Export Error: {e}")
            return False