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
        print(f"[LOG] PromptSeparationWorker 시작. Device: {self.device}")
        
        # 1. 메모리 초기화
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            
        temp_dir = tempfile.mkdtemp()
        try:
            self.progress.emit(5, f"초기화 중... ({self.device})")
            
            try:
                from sam_audio import SAMAudio, SAMAudioProcessor
                # BatchFeature 등 복잡한 객체 사용 안 함
                print("[LOG] 라이브러리 임포트 성공")
            except ImportError:
                 raise ImportError("Meta SAM Audio 라이브러리가 필요합니다.")

            model_id = "facebook/sam-audio-small"
            
            # 2. 프로세서/모델 로드
            print(f"[LOG] 모델 로드 중 (ID: {model_id})...")
            processor = SAMAudioProcessor.from_pretrained(model_id)
            model = SAMAudio.from_pretrained(model_id)
            print("[LOG] 모델 로드 완료.")
            
            # GPU/FP16 설정
            if self.device == "cuda":
                print("[LOG] CUDA(FP16) 설정 적용 중...")
                model = model.half().to(self.device)
                target_dtype = torch.float16
            else:
                print("[LOG] CPU(FP32) 설정 적용 중...")
                model = model.to(self.device)
                target_dtype = torch.float32

            model.eval()

            if self.isInterruptionRequested(): return

            # 3. 청크 설정
            chunk_seconds = 10 
            chunk_samples = int(chunk_seconds * self.sr)
            total_frames = len(self.input_data)
            total_chunks = math.ceil(total_frames / chunk_samples)
            
            final_target_parts = []
            final_residual_parts = []
            
            model_sr = processor.audio_sampling_rate 

            print(f"[LOG] 총 {total_chunks}개 구간 처리 시작. (Input: {self.sr}Hz -> Model: {model_sr}Hz 리샘플링 예정)")
            self.progress.emit(10, f"작업 시작: 총 {total_chunks}개 구간 처리.")

            # [FIX] 모델 입력을 위한 단순 클래스 정의 (호환성 문제 해결)
            class SimpleBatch:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            for i in range(total_chunks):
                if self.isInterruptionRequested(): 
                    print("[LOG] 중단 요청됨.")
                    return
                
                progress_pct = 10 + int((i / total_chunks) * 85)
                self.progress.emit(progress_pct, f"구간 처리 중 ({i+1}/{total_chunks})...")

                # 4. 오디오 자르기
                start = i * chunk_samples
                end = min(start + chunk_samples, total_frames)
                chunk_audio = self.input_data[start:end]
                expected_len = end - start 

                # 임시 파일 저장
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                sf.write(chunk_path, chunk_audio, self.sr)

                # 5. Processor 실행 (데이터 읽기 전용)
                raw_inputs = processor(audios=[chunk_path], descriptions=[self.prompt])

                # =========================================================
                # [FINAL FIX] 단순 객체(SimpleBatch)에 데이터 직접 이식
                # =========================================================
                
                # 5-1. Processor 결과에서 데이터 추출 (딕셔너리화)
                data_map = {}
                # Processor가 반환하는 객체의 속성을 안전하게 추출
                if hasattr(raw_inputs, 'data') and isinstance(raw_inputs.data, dict):
                    data_map.update(raw_inputs.data)
                elif isinstance(raw_inputs, dict):
                    data_map.update(raw_inputs)
                else:
                    # 알 수 없는 객체일 경우 강제 추출
                    try: data_map.update(vars(raw_inputs))
                    except: pass
                    # 필수 키 수동 확인
                    for key in ['input_values', 'pixel_values', 'audios', 'input_features', 'input_ids', 'attention_mask']:
                        if hasattr(raw_inputs, key):
                            data_map[key] = getattr(raw_inputs, key)

                # 5-2. SimpleBatch 객체 생성 및 텐서 변환
                clean_inputs = SimpleBatch()
                
                audio_tensor = None
                tensor_key_name = None
                
                for k, v in data_map.items():
                    if k == 'original_sizes': continue # 수동 설정을 위해 제외
                    
                    # 리스트 -> 텐서
                    if isinstance(v, list):
                        if len(v) > 0 and isinstance(v[0], torch.Tensor):
                            v = torch.stack(v)
                        else:
                            try: v = torch.tensor(v)
                            except: pass
                    elif isinstance(v, np.ndarray):
                        v = torch.from_numpy(v)
                    
                    # GPU/FP16 이동
                    if isinstance(v, torch.Tensor):
                        if k in ['input_values', 'audios', 'input_features']:
                            audio_tensor = v
                            tensor_key_name = k
                            
                        v = v.to(self.device)
                        if v.is_floating_point():
                            v = v.to(dtype=target_dtype)
                    
                    # 객체 속성으로 주입
                    setattr(clean_inputs, k, v)

                # 5-3. [핵심] 오디오 텐서 차원 보정 및 Original Sizes 주입
                # audio_tensor가 발견되지 않았을 경우 대비
                if audio_tensor is None:
                    # clean_inputs에서 다시 탐색
                    for k in ['input_values', 'audios', 'input_features']:
                        if hasattr(clean_inputs, k):
                            audio_tensor = getattr(clean_inputs, k)
                            tensor_key_name = k
                            break
                
                if audio_tensor is not None:
                    # GPU에 올라간 최신 텐서 가져오기
                    curr_tensor = getattr(clean_inputs, tensor_key_name)
                    
                    # 실제 길이 계산 (Time Dimension)
                    real_len = curr_tensor.shape[-1]
                    
                    # (1) 차원 보정: (Batch, Time) -> (Batch, 1, Time)
                    if curr_tensor.ndim == 2:
                        curr_tensor = curr_tensor.unsqueeze(1)
                        setattr(clean_inputs, tensor_key_name, curr_tensor)
                    
                    # (2) [오류 해결] original_sizes를 '파이썬 정수 리스트'로 주입
                    # 텐서([48000]) 대신 리스트 [48000]을 주면 모델이 확실하게 읽습니다.
                    setattr(clean_inputs, 'original_sizes', [int(real_len)])
                    
                    # (3) 안전장치: attention_mask 생성
                    if not hasattr(clean_inputs, 'attention_mask'):
                         mask = torch.ones((1, real_len), device=self.device, dtype=torch.long)
                         setattr(clean_inputs, 'attention_mask', mask)
                else:
                    print(f"[WARN] [{i+1}] 오디오 텐서를 찾지 못했습니다.")

                # 6. 추론
                with torch.no_grad():
                    # 이제 clean_inputs는 완벽한 파이썬 객체입니다.
                    outputs = model.separate(clean_inputs)
                
                # 7. 결과 추출
                if outputs.target is not None and len(outputs.target) > 0:
                    gen_tensor = outputs.target[0].float().detach().cpu()
                    t_wav = gen_tensor.numpy()
                    if t_wav.ndim == 2 and t_wav.shape[0] == 1: t_wav = np.squeeze(t_wav)
                else:
                    t_wav = np.zeros((int(chunk_seconds * model_sr),), dtype=np.float32)

                if outputs.residual is not None and len(outputs.residual) > 0:
                    res_tensor = outputs.residual[0].float().detach().cpu()
                    r_wav = res_tensor.numpy()
                    if r_wav.ndim == 2 and r_wav.shape[0] == 1: r_wav = np.squeeze(r_wav)
                else:
                    r_wav = np.zeros_like(t_wav)

                # 8. 리샘플링
                if model_sr != self.sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=model_sr, new_freq=self.sr)
                    t_tensor_in = torch.from_numpy(t_wav)
                    if t_tensor_in.ndim == 1: t_tensor_in = t_tensor_in.unsqueeze(0)
                    r_tensor_in = torch.from_numpy(r_wav)
                    if r_tensor_in.ndim == 1: r_tensor_in = r_tensor_in.unsqueeze(0)

                    t_wav = resampler(t_tensor_in).squeeze().numpy()
                    r_wav = resampler(r_tensor_in).squeeze().numpy()

                # 9. 길이 및 채널 맞춤
                t_part = ensure_stereo_np(t_wav.T if t_wav.ndim > 1 else t_wav)
                r_part = ensure_stereo_np(r_wav.T if r_wav.ndim > 1 else r_wav)
                
                if len(t_part) > expected_len:
                    t_part = t_part[:expected_len]
                elif len(t_part) < expected_len:
                    pad = np.zeros((expected_len - len(t_part), 2), dtype=np.float32)
                    t_part = np.vstack([t_part, pad])

                if len(r_part) > expected_len:
                    r_part = r_part[:expected_len]
                elif len(r_part) < expected_len:
                    pad = np.zeros((expected_len - len(r_part), 2), dtype=np.float32)
                    r_part = np.vstack([r_part, pad])

                final_target_parts.append(t_part)
                final_residual_parts.append(r_part)

                del raw_inputs, clean_inputs, outputs, data_map
                try: os.remove(chunk_path) 
                except: pass
                
                if self.device == "cuda": 
                    torch.cuda.empty_cache()

            print("[LOG] 모든 구간 처리 완료. 병합 중...")
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
                print(f"[ERROR] SAM Audio 오류: {e}")
                import traceback; traceback.print_exc()
                self.failed.emit(f"SAM Audio 오류: {str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("[LOG] 종료.")