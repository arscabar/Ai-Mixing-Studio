import os
import sys
import shutil
import warnings
import numpy as np
import tempfile
import math
import gc
from PyQt6.QtCore import QThread, pyqtSignal

# =========================================================================
# [FFmpeg 경로 설정]
# =========================================================================
ffmpeg_dll_path = r"E:\ffmpeg\bin"
if os.path.exists(ffmpeg_dll_path):
    os.environ["TORCHCODEC_FFMPEG_DIR"] = ffmpeg_dll_path
    os.environ["PATH"] = ffmpeg_dll_path + os.pathsep + os.environ["PATH"]
    try:
        os.add_dll_directory(ffmpeg_dll_path)
    except Exception:
        pass

import torch
import torchaudio
from transformers import pipeline
import whisper
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")

from .models import Track, SourceType, SubtitleItem
from .audio_io import ensure_stereo_np
from .media_util import is_video_file
from .ffmpeg_util import extract_audio_to_wav, find_ffmpeg

AUDIO_EXTS = {".wav", ".flac", ".aiff", ".aif"}

def _ext(p: str) -> str:
    return os.path.splitext(p)[1].lower()

def _need_ffmpeg_decode(path: str) -> bool:
    if is_video_file(path): return True
    return _ext(path) not in AUDIO_EXTS

def _load_audio_numpy_safe(path: str, sr: int) -> np.ndarray:
    if _need_ffmpeg_decode(path):
        if not find_ffmpeg():
            import librosa
            y, _ = librosa.load(path, sr=sr, mono=False)
            if y.ndim == 1: y = np.stack([y, y], axis=0)
            return ensure_stereo_np(y.T)

        td = tempfile.TemporaryDirectory()
        wav_path = os.path.join(td.name, "decoded.wav")
        try:
            extract_audio_to_wav(path, wav_path)
            audio, file_sr = sf.read(wav_path, dtype="float32", always_2d=True)
            if file_sr != sr:
                w = torch.from_numpy(audio.T)
                res = torchaudio.transforms.Resample(file_sr, sr)(w)
                audio = res.T.contiguous().cpu().numpy()
            return ensure_stereo_np(audio)
        except Exception:
            import librosa
            y, _ = librosa.load(path, sr=sr, mono=False)
            if y.ndim == 1: y = np.stack([y, y], axis=0)
            return ensure_stereo_np(y.T)
        finally:
            td.cleanup()

    try:
        audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
        if file_sr != sr:
            w = torch.from_numpy(audio.T)
            res = torchaudio.transforms.Resample(file_sr, sr)(w)
            audio = res.T.contiguous().cpu().numpy()
        return ensure_stereo_np(audio)
    except Exception:
        import librosa
        y, _ = librosa.load(path, sr=sr, mono=False)
        if y.ndim == 1: y = np.stack([y, y], axis=0)
        return ensure_stereo_np(y.T)


class AISeparationWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str, user_sr: int = 44100, device_pref: str = "auto"):
        super().__init__()
        self.file_path = file_path
        self.user_sr = int(user_sr)
        self.device = "cuda" if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu"

    def _check_model_cached(self) -> bool:
        try:
            hub_dir = torch.hub.get_dir()
            for root, dirs, files in os.walk(hub_dir):
                for f in files:
                    if "hdemucs" in f and ".pt" in f: return True
            return False
        except: return False

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
                classifier = pipeline("audio-classification", model="mit/ast-finetuned-audioset-10-10-0.4593", device=0 if self.device == "cuda" else -1)
                tags = classifier(self.file_path, top_k=3)
                detected = [t["label"] for t in tags]
                labels_str = ", ".join(detected)
                self.progress.emit(25, f"감지됨: {labels_str}")
            except Exception:
                self.progress.emit(25, "Tagging 실패 (건너뜀)")

            if self.isInterruptionRequested(): return

            is_cached = self._check_model_cached()
            msg = "AI: HDemucs 모델 로드 중..." if is_cached else "AI: 모델 다운로드 중 (약 3GB)..."
            self.progress.emit(35, msg)
            
            bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
            model = bundle.get_model().to(self.device)
            model.eval()

            self.progress.emit(45, f"AI: 전처리... ({self.device})")
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
                    
                    if self.device == "cuda": torch.cuda.empty_cache()

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
                if len(src) > target_len: src = src[:target_len]
                elif len(src) < target_len:
                    pad = np.zeros((target_len - len(src), 2), dtype=np.float32)
                    src = np.vstack([src, pad])

                display = f"{name} (AI)"
                if name == "Vocals" and ("Speech" in labels_str or "speech" in labels_str):
                    display = "Speech/Vocals (AI)"

                t = Track(display, src, sr=self.user_sr, source_type=SourceType.AI_SEPARATED)
                tracks.append(t)
                sum_sep += src

            residual = y_user.astype(np.float32, copy=False) - sum_sep
            tracks.append(Track("Residual (Ambience)", residual, sr=self.user_sr, source_type=SourceType.RESIDUAL))

            self.progress.emit(100, f"완료 ({labels_str})")
            self.finished.emit(tracks)

        except Exception as e:
            if not self.isInterruptionRequested():
                import traceback; traceback.print_exc()
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
            name = os.path.splitext(os.path.basename(self.file_path))[0]
            track = Track(f"{name} (Ext)", y, sr=self.sr, source_type=SourceType.IMPORTED)
            self.finished.emit(track)
        except Exception as e:
            self.failed.emit(str(e))

class SubtitleGenerationWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, audio_path: str, model_size: str = "small", device_pref: str = "cuda"):
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        self.device = "cuda" if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu"

    def run(self):
        try:
            self.progress.emit(10, f"Whisper 모델 로드 ({self.model_size}, {self.device})...")
            model = whisper.load_model(self.model_size, device=self.device)

            self.progress.emit(30, "오디오 인식 및 자막 생성 중...")
            use_fp16 = (self.device == "cuda")
            result = model.transcribe(self.audio_path, fp16=use_fp16)

            detected_lang = result.get('language', 'unknown')
            self.progress.emit(80, f"언어 감지: {detected_lang}, 데이터 변환 중...")

            subtitles = []
            for segment in result['segments']:
                item = SubtitleItem(
                    start_time=segment['start'],
                    end_time=segment['end'],
                    text=segment['text'].strip(),
                    language=detected_lang
                )
                subtitles.append(item)

            self.progress.emit(100, "자막 생성 완료!")
            self.finished.emit(subtitles)

        except Exception as e:
            import traceback; traceback.print_exc()
            self.failed.emit(str(e))

# =========================================================================
# [최종 수정] SAM Audio 워커 (Tensor original_sizes 적용 + MutableBatch)
# =========================================================================

# 모델에게 전달할 데이터 컨테이너 (dict + attr 접근 가능)
class MutableBatch(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    
    def __setattr__(self, name, value):
        self[name] = value

class PromptSeparationWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, input_track_data: np.ndarray, sr: int, prompt: str, device_pref: str = "cuda"):
        super().__init__()
        self.input_data = input_track_data 
        self.sr = sr
        self.prompt = prompt
        self.device = "cuda" if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu"

    def run(self):
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            
        temp_dir = tempfile.mkdtemp()
        try:
            self.progress.emit(5, f"초기화 중... ({self.device})")
            
            try:
                from sam_audio import SAMAudio, SAMAudioProcessor
            except ImportError:
                 raise ImportError("Meta SAM Audio 라이브러리가 필요합니다.")

            model_id = "facebook/sam-audio-small"
            
            # 1. 프로세서/모델 로드
            processor = SAMAudioProcessor.from_pretrained(model_id)
            model = SAMAudio.from_pretrained(model_id)
            
            # GPU/FP16 설정
            if self.device == "cuda":
                model = model.half().to(self.device)
                target_dtype = torch.float16
            else:
                model = model.to(self.device)
                target_dtype = torch.float32

            model.eval()

            if self.isInterruptionRequested(): return

            # 청크 설정 (10초)
            chunk_seconds = 10 
            chunk_samples = int(chunk_seconds * self.sr)
            total_frames = len(self.input_data)
            total_chunks = math.ceil(total_frames / chunk_samples)
            
            final_target_parts = []
            final_residual_parts = []
            
            model_sr = processor.audio_sampling_rate 

            self.progress.emit(10, f"작업 시작: 총 {total_chunks}개 구간.")

            for i in range(total_chunks):
                if self.isInterruptionRequested(): return
                
                progress_pct = 10 + int((i / total_chunks) * 85)
                self.progress.emit(progress_pct, f"구간 처리 중 ({i+1}/{total_chunks})...")

                start = i * chunk_samples
                end = min(start + chunk_samples, total_frames)
                chunk_audio = self.input_data[start:end]
                
                # 임시 저장
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                sf.write(chunk_path, chunk_audio, self.sr)

                # 2. 프로세서로부터 원본 데이터 받기
                raw_inputs = processor(audios=[chunk_path], descriptions=[self.prompt])

                # =========================================================
                # [해결책] MutableBatch + Tensor original_sizes 주입
                # =========================================================
                clean_inputs = MutableBatch()

                # 2-1. 원본 오디오 텐서 추출 (BatchFeature 처리)
                # raw_inputs가 BatchFeature 객체일 경우 .data 속성을, 
                # 딕셔너리일 경우 그 자체를 사용하여 값 추출
                if hasattr(raw_inputs, 'data'):
                    data_source = raw_inputs.data
                else:
                    data_source = raw_inputs

                # audios 가져오기
                if 'audios' in data_source:
                    audio_tensor = data_source['audios'][0]
                elif hasattr(raw_inputs, 'audios'):
                    audio_tensor = raw_inputs.audios[0]
                else:
                    # 키가 없으면 values() 첫번째로 시도 (fallback)
                    audio_tensor = list(data_source.values())[0][0]

                real_len_16k = audio_tensor.shape[-1]
                
                # (1) Audio Tensor: GPU/FP16 이동 + 배치 차원(1) 추가
                gpu_tensor = audio_tensor.to(self.device, dtype=target_dtype)
                if gpu_tensor.ndim == 2:
                    gpu_tensor = gpu_tensor.unsqueeze(0) # (1, C, T)
                
                clean_inputs.audios = gpu_tensor
                
                # (2) [핵심] Original Sizes를 'LongTensor'로 주입
                # 리스트([]) 대신 텐서로 넣어야 narrow 연산 시 에러 안 남
                clean_inputs.original_sizes = torch.LongTensor([real_len_16k])
                
                # (3) 나머지 속성 복사 (input_ids, attention_mask 등)
                for k, v in data_source.items():
                    if k in ['audios', 'original_sizes']: continue 
                    
                    # 텐서만 변환 (정수형은 FP16 변환 금지, GPU 이동만)
                    if isinstance(v, torch.Tensor):
                        v = v.to(self.device)
                        if v.is_floating_point():
                            v = v.to(dtype=target_dtype)
                    
                    setattr(clean_inputs, k, v)

                # 3. 추론
                with torch.no_grad():
                    outputs = model.separate(clean_inputs)
                
                # 4. 결과 추출
                if len(outputs.target) > 0:
                    gen_tensor = outputs.target[0].float().detach().cpu() 
                    t_wav = gen_tensor.numpy()
                    t_wav = np.squeeze(t_wav)
                else:
                    t_wav = np.zeros((real_len_16k,), dtype=np.float32)

                if outputs.residual is not None and len(outputs.residual) > 0:
                    res_tensor = outputs.residual[0].float().detach().cpu()
                    r_wav = res_tensor.numpy()
                    r_wav = np.squeeze(r_wav)
                else:
                    r_wav = np.zeros_like(t_wav)

                # 5. 리샘플링
                if model_sr != self.sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=model_sr, new_freq=self.sr)
                    
                    t_tensor_in = torch.from_numpy(t_wav)
                    if t_tensor_in.ndim == 1: t_tensor_in = t_tensor_in.unsqueeze(0)
                    r_tensor_in = torch.from_numpy(r_wav)
                    if r_tensor_in.ndim == 1: r_tensor_in = r_tensor_in.unsqueeze(0)

                    t_wav = resampler(t_tensor_in).squeeze().numpy()
                    r_wav = resampler(r_tensor_in).squeeze().numpy()

                # 6. Stereo & 길이 맞춤
                t_part = ensure_stereo_np(t_wav.T)
                r_part = ensure_stereo_np(r_wav.T)
                
                expected_len = end - start 
                
                if len(t_part) > expected_len: t_part = t_part[:expected_len]
                if len(r_part) > expected_len: r_part = r_part[:expected_len]
                
                if len(t_part) < expected_len:
                    pad = np.zeros((expected_len - len(t_part), 2), dtype=np.float32)
                    t_part = np.vstack([t_part, pad])
                    r_part = np.vstack([r_part, pad])

                final_target_parts.append(t_part)
                final_residual_parts.append(r_part)

                del raw_inputs, clean_inputs, outputs, gen_tensor, t_wav, r_wav
                try: os.remove(chunk_path) 
                except: pass
                if self.device == "cuda": torch.cuda.empty_cache()

            self.progress.emit(95, "결과 병합 중...")
            target_data = np.concatenate(final_target_parts, axis=0)
            residual_data = np.concatenate(final_residual_parts, axis=0)

            tracks = []
            stype = getattr(SourceType, 'PROMPT_SEPARATED', SourceType.AI_SEPARATED)
            tracks.append(Track(f"{self.prompt} (SAM)", target_data, sr=self.sr, source_type=stype))
            tracks.append(Track(f"Residue (w/o {self.prompt})", residual_data, sr=self.sr, source_type=SourceType.RESIDUAL))

            self.progress.emit(100, "완료")
            self.finished.emit(tracks)

            del model, processor
            if self.device == "cuda": torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            if not self.isInterruptionRequested():
                import traceback; traceback.print_exc()
                self.failed.emit(f"SAM Audio 오류: {str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)