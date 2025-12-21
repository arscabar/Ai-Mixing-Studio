# src/ai_service.py
import os
import sys
import warnings
import numpy as np
import tempfile
import math
from PyQt6.QtCore import QThread, pyqtSignal

import torch
import torchaudio
from transformers import pipeline

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")

from .models import Track, SourceType
from .audio_io import ensure_stereo_np
from .media_util import is_video_file
from .ffmpeg_util import extract_audio_to_wav, find_ffmpeg

AUDIO_EXTS = {".wav", ".flac", ".aiff", ".aif"}

def _ext(p: str) -> str:
    return os.path.splitext(p)[1].lower()

def _need_ffmpeg_decode(path: str) -> bool:
    if is_video_file(path):
        return True
    return _ext(path) not in AUDIO_EXTS

def _load_audio_numpy_safe(path: str, sr: int) -> np.ndarray:
    if _need_ffmpeg_decode(path):
        if not find_ffmpeg():
            import librosa
            y, _ = librosa.load(path, sr=sr, mono=False)
            if y.ndim == 1:
                y = np.stack([y, y], axis=0)
            return ensure_stereo_np(y.T)

        td = tempfile.TemporaryDirectory()
        wav_path = os.path.join(td.name, "decoded.wav")
        extract_audio_to_wav(path, wav_path)
        try:
            import soundfile as sf
            audio, file_sr = sf.read(wav_path, dtype="float32", always_2d=True)
            if file_sr != sr:
                w = torch.from_numpy(audio.T)
                res = torchaudio.transforms.Resample(file_sr, sr)(w)
                audio = res.T.contiguous().cpu().numpy()
            return ensure_stereo_np(audio)
        finally:
            td.cleanup()

    import soundfile as sf
    audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if file_sr != sr:
        w = torch.from_numpy(audio.T)
        res = torchaudio.transforms.Resample(file_sr, sr)(w)
        audio = res.T.contiguous().cpu().numpy()
    return ensure_stereo_np(audio)


class AISeparationWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str, user_sr: int = 44100, device_pref: str = "auto"):
        super().__init__()
        self.file_path = file_path
        self.user_sr = int(user_sr)
        if device_pref == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def _check_model_cached(self) -> bool:
        try:
            hub_dir = torch.hub.get_dir()
            for root, dirs, files in os.walk(hub_dir):
                for f in files:
                    if "hdemucs" in f and ".pt" in f:
                        return True
            return False
        except:
            return False

    def run(self):
        try:
            tracks = []

            if self.isInterruptionRequested(): return

            self.progress.emit(5, "오디오 로드 중...")
            y_user = _load_audio_numpy_safe(self.file_path, sr=self.user_sr)

            if self.isInterruptionRequested(): return

            labels_str = "Unknown"
            self.progress.emit(15, "AI: 소리 성분 분석 중 (Tagging)...")
            try:
                classifier = pipeline(
                    "audio-classification",
                    model="mit/ast-finetuned-audioset-10-10-0.4593",
                    device=0 if self.device == "cuda" else -1,
                )
                tags = classifier(self.file_path, top_k=3)
                detected = [t["label"] for t in tags]
                labels_str = ", ".join(detected)
                self.progress.emit(25, f"감지됨: {labels_str}")
            except Exception as e:
                self.progress.emit(25, "Tagging 실패")
                print(f"Tagging Warning: {e}")

            if self.isInterruptionRequested(): return

            is_cached = self._check_model_cached()
            msg = "AI: HDemucs 모델 로드 중..."
            if not is_cached:
                msg = "AI: 모델 다운로드 중 (약 3GB)... 잠시만 기다려주세요."
            self.progress.emit(35, msg)
            
            bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
            model = bundle.get_model().to(self.device)
            model.eval()

            self.progress.emit(45, f"AI: 데이터 전처리... (device={self.device})")

            waveform = torch.from_numpy(y_user.T).to(self.device)
            
            if self.user_sr != bundle.sample_rate:
                resampler = torchaudio.transforms.Resample(self.user_sr, bundle.sample_rate).to(self.device)
                waveform = resampler(waveform)

            ref = waveform.mean()
            std = waveform.std().clamp_min(1e-5)
            waveform_n = (waveform - ref) / std

            sr = bundle.sample_rate
            chunk_seconds = 30
            chunk_size = int(sr * chunk_seconds)
            total_samples = waveform_n.shape[1]
            total_chunks = math.ceil(total_samples / chunk_size)

            separated_chunks = []

            with torch.no_grad():
                for i in range(total_chunks):
                    if self.isInterruptionRequested(): return
                    
                    start = i * chunk_size
                    end = min(start + chunk_size, total_samples)
                    
                    progress_pct = 50 + int((i / total_chunks) * 40)
                    self.progress.emit(progress_pct, f"AI: 분리 중 ({i+1}/{total_chunks})...")

                    chunk_in = waveform_n[:, start:end]
                    sources_chunk = model(chunk_in[None]) 
                    separated_chunks.append(sources_chunk[0].detach().cpu())
                    
                    del sources_chunk, chunk_in
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

            self.progress.emit(90, "AI: 트랙 병합 중...")
            
            sources = torch.cat(separated_chunks, dim=-1)
            sources = sources * std.cpu() + ref.cpu()
            sources = sources.numpy()

            self.progress.emit(95, "트랙 생성 중...")

            target_len = y_user.shape[0]
            source_names = ["Drums", "Bass", "Other", "Vocals"]
            sum_sep = np.zeros_like(y_user, dtype=np.float32)

            for i, name in enumerate(source_names):
                if self.isInterruptionRequested(): return

                src = ensure_stereo_np(sources[i].T)
                if len(src) > target_len:
                    src = src[:target_len]
                elif len(src) < target_len:
                    pad = np.zeros((target_len - len(src), 2), dtype=np.float32)
                    src = np.vstack([src, pad])

                display = f"{name} (AI)"
                if name == "Vocals" and ("Speech" in labels_str or "speech" in labels_str):
                    display = "Speech/Vocals (AI)"

                t = Track(display, src, sr=self.user_sr, source_type=SourceType.AI_SEPARATED)
                tracks.append(t)
                sum_sep += src

            if self.isInterruptionRequested(): return

            residual = y_user.astype(np.float32, copy=False) - sum_sep
            tracks.append(Track("Residual (Ambience)", residual, sr=self.user_sr, source_type=SourceType.RESIDUAL))

            self.progress.emit(100, f"완료 ({labels_str})")
            self.finished.emit(tracks)

        except Exception as e:
            if not self.isInterruptionRequested():
                import traceback
                traceback.print_exc()
                self.failed.emit(str(e))


class ExternalFileLoader(QThread):
    finished = pyqtSignal(Track)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str, sr: int = 44100):
        super().__init__()
        self.file_path = file_path
        self.sr = int(sr)

    def run(self):
        try:
            if self.isInterruptionRequested(): return
            y = _load_audio_numpy_safe(self.file_path, sr=self.sr)
            if self.isInterruptionRequested(): return
            name = os.path.splitext(os.path.basename(self.file_path))[0]
            track = Track(f"{name} (Ext)", y, sr=self.sr, source_type=SourceType.IMPORTED)
            self.finished.emit(track)
        except Exception as e:
            if not self.isInterruptionRequested():
                self.failed.emit(str(e))