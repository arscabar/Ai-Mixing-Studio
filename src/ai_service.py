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
# [수정됨] SAM Audio 워커 - batch.sizes 문제 해결
# =========================================================================

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
            
            # model.sample_rate 사용 (audio_codec의 sample_rate)
            model_sr = model.sample_rate
            print(f"[LOG] 모델 sample_rate: {model_sr}")

            print(f"[LOG] 총 {total_chunks}개 구간 처리 시작. (Input: {self.sr}Hz -> Model: {model_sr}Hz)")
            self.progress.emit(10, f"작업 시작: 총 {total_chunks}개 구간 처리.")

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

                # 5. Processor 실행
                batch = processor(audios=[chunk_path], descriptions=[self.prompt])

                # =========================================================
                # [핵심 수정] batch.sizes 계산 및 주입
                # model.separate()가 batch.sizes를 읽어서 unbatch()에 전달함
                # sizes가 없으면 feature_idx_to_wav_idx(None) -> INT_MAX 오류
                # =========================================================
                
                # 5-1. 오디오 텐서 찾기 및 길이 계산
                audio_tensor = None
                if hasattr(batch, 'audios') and batch.audios is not None:
                    audio_tensor = batch.audios
                
                if audio_tensor is None:
                    raise ValueError(f"[{i+1}] batch에서 audios를 찾을 수 없습니다")
                
                # 리스트인 경우 첫 번째 요소 사용
                if isinstance(audio_tensor, list):
                    if len(audio_tensor) > 0 and isinstance(audio_tensor[0], torch.Tensor):
                        audio_tensor = audio_tensor[0]
                    else:
                        audio_tensor = torch.tensor(audio_tensor)
                
                # 오디오 샘플 수 (model_sr 기준)
                audio_length_samples = audio_tensor.shape[-1]
                print(f"[LOG] [{i+1}] audio_tensor shape: {audio_tensor.shape}, samples: {audio_length_samples}")
                
                # 5-2. feature 길이 계산 (codec의 downsample ratio)
                # DAC codec은 보통 hop_length=512 사용
                try:
                    if hasattr(model, 'audio_codec'):
                        codec = model.audio_codec
                        if hasattr(codec, 'hop_length'):
                            hop = codec.hop_length
                        elif hasattr(codec, 'downsample_rate'):
                            hop = codec.downsample_rate
                        elif hasattr(codec, 'encoder') and hasattr(codec.encoder, 'hop_length'):
                            hop = codec.encoder.hop_length
                        else:
                            hop = 512  # DAC 기본값
                    else:
                        hop = 512
                    
                    feature_length = audio_length_samples // hop
                    print(f"[LOG] [{i+1}] hop: {hop}, feature_length: {feature_length}")
                except Exception as e:
                    print(f"[LOG] [{i+1}] hop 계산 실패: {e}, 기본값 512 사용")
                    feature_length = audio_length_samples // 512
                
                # 5-3. sizes 텐서 생성 및 주입
                # model.separate() 내부:
                #   sizes = self.audio_codec.feature_idx_to_wav_idx(batch.sizes)
                # 이므로 feature 인덱스 기준 길이를 전달해야 함
                sizes_tensor = torch.tensor([feature_length], dtype=torch.long, device=self.device)
                
                # batch 객체에 sizes 주입 (여러 방식 시도)
                batch.sizes = sizes_tensor
                
                # 호환성을 위해 data 딕셔너리에도 설정
                if hasattr(batch, 'data') and isinstance(batch.data, dict):
                    batch.data['sizes'] = sizes_tensor
                
                print(f"[LOG] [{i+1}] 주입된 sizes: {sizes_tensor}")

                # 5-4. GPU로 이동
                batch = batch.to(self.device)
                
                # sizes가 GPU로 이동했는지 확인
                if hasattr(batch, 'sizes') and isinstance(batch.sizes, torch.Tensor):
                    if batch.sizes.device.type != self.device:
                        batch.sizes = batch.sizes.to(self.device)

                # 5-5. FP16 변환 (오디오 텐서)
                if target_dtype == torch.float16:
                    if hasattr(batch, 'audios'):
                        if isinstance(batch.audios, torch.Tensor):
                            batch.audios = batch.audios.half()
                        elif isinstance(batch.audios, list):
                            batch.audios = [a.half() if isinstance(a, torch.Tensor) else a for a in batch.audios]

                # 6. 추론
                with torch.no_grad():
                    outputs = model.separate(batch)
                
                # 7. 결과 추출
                if outputs.target is not None and len(outputs.target) > 0:
                    gen_tensor = outputs.target[0].float().detach().cpu()
                    t_wav = gen_tensor.numpy()
                    if t_wav.ndim == 2 and t_wav.shape[0] == 1: 
                        t_wav = np.squeeze(t_wav)
                    print(f"[LOG] [{i+1}] target shape: {t_wav.shape}")
                else:
                    t_wav = np.zeros((audio_length_samples,), dtype=np.float32)
                    print(f"[LOG] [{i+1}] target 없음, 0으로 채움")

                if outputs.residual is not None and len(outputs.residual) > 0:
                    res_tensor = outputs.residual[0].float().detach().cpu()
                    r_wav = res_tensor.numpy()
                    if r_wav.ndim == 2 and r_wav.shape[0] == 1: 
                        r_wav = np.squeeze(r_wav)
                else:
                    r_wav = np.zeros_like(t_wav)

                # 8. 리샘플링 (model_sr -> self.sr)
                if model_sr != self.sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=model_sr, new_freq=self.sr)
                    t_tensor_in = torch.from_numpy(t_wav)
                    if t_tensor_in.ndim == 1: 
                        t_tensor_in = t_tensor_in.unsqueeze(0)
                    r_tensor_in = torch.from_numpy(r_wav)
                    if r_tensor_in.ndim == 1: 
                        r_tensor_in = r_tensor_in.unsqueeze(0)

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

                # 메모리 정리
                del batch, outputs
                try: 
                    os.remove(chunk_path) 
                except Exception: 
                    pass
                
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
            if self.device == "cuda": 
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            if not self.isInterruptionRequested():
                print(f"[ERROR] SAM Audio 오류: {e}")
                import traceback; traceback.print_exc()
                self.failed.emit(f"SAM Audio 오류: {str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("[LOG] 종료.")