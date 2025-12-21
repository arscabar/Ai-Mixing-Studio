# AI Mixing Studio v8 (full)
## 포함 사항
- GPU 사용 가능 시 사용, 불가하면 자동 CPU 폴백
- MP4 입력도 처리: **영상 스트림을 쓰지 않고 오디오만 추출**하여 AI 분리/레이어로 사용
- Export: **믹스 결과 오디오(WAV)만 저장**
- 모델 캐시 포함 배포용 부트스트랩:
  - PyInstaller로 `models_cache_seed` 포함
  - 최초 실행 시 `%APPDATA%\AI_Mixing_Studio\models_cache`로 복사 후 사용
- 로그 저장:
  - `%APPDATA%\AI_Mixing_Studio\logs\app.log` (로테이션)

## 실행 (개발)
```powershell
python -m venv .venv
.\.venv\Scripts\activate

REM CPU PC:
.\install_cpu.bat
REM 또는 CUDA PC:
REM .\install_cuda.bat

python -m src.download_models
python run.py
```

## ffmpeg
- MP4 입력(오디오 추출)에 ffmpeg 권장
- `bin/ffmpeg.exe` 로 넣으면 자동 인식
- 또는 환경변수 `FFMPEG_PATH`로 지정

## 빌드 (PyInstaller, onedir)
```powershell
.\build_onedir_cpu.bat
```
결과: `dist/AI_Mixing_Studio/AI_Mixing_Studio.exe`

##예시 영상

## 추후 예정
볼륨 조절 및 방향과 같은 속성의 변화를 어떤 시간에 각각 적용할것인지도 다 저장하여, 시간의 순서에 맞게 속성의 변화를 작업
텍스트 박스 위치를 드래그로 조정


## Windows 참고 (중요)
- torchaudio 2.9+는 load/save 구현이 TorchCodec 기반으로 바뀌었습니다. 
- TorchCodec은 Windows 지원이 아직 불완전하다는 릴리즈 노트가 있어 `Could not load libtorchcodec`가 발생할 수 있습니다. 
- v9는 **torchaudio.load를 사용하지 않고** numpy 기반 로딩으로 우회해 해당 문제를 피합니다.
