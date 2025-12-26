# src/audio_engine.py
import numpy as np
import sounddevice as sd
import soundfile as sf
from .models import ProjectContext

class AudioEngine:
    def __init__(self, context: ProjectContext):
        self.ctx = context
        self.stream: sd.OutputStream | None = None
        self.vis_buffer: np.ndarray | None = None

    def start(self):
        if self.stream is None:
            self.stream = sd.OutputStream(
                samplerate=self.ctx.sample_rate,
                blocksize=2048,
                channels=2,
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

    # ==========================================================
    #  [추가됨] UI 컨트롤 연동 메서드 (Volume, Mute, Solo)
    # ==========================================================
    def set_track_volume(self, name: str, vol: float):
        for t in self.ctx.tracks:
            if t.name == name:
                t.volume = vol
                break

    def set_track_mute(self, name: str, mute: bool):
        for t in self.ctx.tracks:
            if t.name == name:
                t.mute = mute
                break

    def set_track_solo(self, name: str, solo: bool):
        for t in self.ctx.tracks:
            if t.name == name:
                t.solo = solo
                break
    # ==========================================================

    def _audio_callback(self, outdata, frames, time, status):
        if not self.ctx.is_playing or not self.ctx.tracks:
            outdata.fill(0)
            self.vis_buffer = np.zeros((frames, 2), dtype=np.float32)
            return

        mix_buffer = self._mix_chunk(self.ctx.current_frame, frames)
        
        valid_len = len(mix_buffer)
        outdata[:valid_len] = mix_buffer
        if valid_len < frames:
            outdata[valid_len:].fill(0)
            
        self.vis_buffer = mix_buffer if valid_len == frames else np.pad(mix_buffer, ((0, frames-valid_len), (0,0)))
        self.ctx.current_frame += frames

    def _mix_chunk(self, start_frame: int, n_frames: int) -> np.ndarray:
        mix = np.zeros((n_frames, 2), dtype=np.float32)
        end_frame = start_frame + n_frames
        current_time = start_frame / float(self.ctx.sample_rate)

        # [Logic] Solo가 하나라도 있으면 Solo 아닌 트랙은 무시
        any_solo = any(t.solo for t in self.ctx.tracks)

        for t in self.ctx.tracks:
            # 1. Mute/Solo Check
            if t.mute: continue
            if any_solo and not t.solo: continue # Solo 모드인데 Solo 아니면 스킵
            
            t_len = len(t.data)
            if start_frame >= t_len: continue
            eff_end = min(end_frame, t_len)
            eff_len = eff_end - start_frame
            chunk = t.data[start_frame:eff_end]
            
            if eff_len < n_frames:
                chunk = np.pad(chunk, ((0, n_frames - eff_len), (0, 0)))

            # Volume (Automation + Base Volume)
            vol = t.get_automated_value("volume", current_time, t.volume)
            y = chunk * vol
            
            # Pan (Automation + Base Pan)
            pan_angle = t.get_automated_value("pan", current_time, t.angle_deg)
            y = self._apply_panning(y, pan_angle)
            
            # EQ
            y *= (t.eq_low + t.eq_mid + t.eq_high) / 3.0
            
            mix += y

        mix *= self.ctx.master_volume
        return np.tanh(mix).astype(np.float32)

    def _apply_panning(self, audio: np.ndarray, angle_deg: float) -> np.ndarray:
        rad = np.radians(angle_deg % 360)
        pan_val = np.sin(rad) 
        theta = (pan_val + 1.0) * (np.pi / 4.0)
        gain_l = np.cos(theta)
        gain_r = np.sin(theta)
        out = audio.copy()
        out[:, 0] *= gain_l
        out[:, 1] *= gain_r
        return out