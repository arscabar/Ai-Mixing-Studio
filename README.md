🎛️ AI Mixing Studio v8 (Full Version)
AI Mixing Studio는 딥러닝 기반의 음원 분리 및 믹싱 도구입니다. GPU 가속을 지원하며, MP4 영상 파일에서 오디오를 추출하여 작업할 수 있습니다.

✨ 주요 기능 (Key Features)
⚡ 스마트 하드웨어 가속

NVIDIA GPU 사용 가능 시 자동 활성화 (CUDA).

GPU가 없거나 사용 불가능할 경우 자동으로 CPU 모드로 전환됩니다.

🎥 미디어 호환성 (MP4 지원)

영상 파일(MP4) 입력 시, 영상 스트림은 제외하고 오디오 트랙만 자동으로 추출하여 AI 분리 및 레이어 작업에 사용합니다.

💾 내보내기 (Export)

최종 믹싱 결과물은 고품질 오디오(WAV) 파일로 저장됩니다.

🚀 배포 최적화 (Bootstrap)

PyInstaller 빌드 시 models_cache_seed를 포함하여 패키징합니다.

사용자 PC에서 최초 실행 시, 모델 파일을 %APPDATA%\AI_Mixing_Studio\models_cache로 자동 복사하여 초기화합니다.

📝 로깅 시스템

애플리케이션 로그는 다음 경로에 로테이션 방식으로 저장됩니다.

경로: %APPDATA%\AI_Mixing_Studio\logs\app.log

🛠️ 개발 환경 설정 및 실행 (Development)
개발 환경을 세팅하고 애플리케이션을 실행하는 방법입니다.

1. 가상 환경 구성
PowerShell

python -m venv .venv
.\.venv\Scripts\activate
2. 의존성 패키지 설치
PC 환경에 맞는 스크립트를 실행해 주세요.

CPU 전용 PC:

PowerShell

.\install_cpu.bat
NVIDIA GPU (CUDA) PC:

PowerShell

.\install_cuda.bat
3. 모델 다운로드 및 실행
필요한 AI 모델을 다운로드한 후 프로그램을 시작합니다.

PowerShell

python -m src.download_models
python run.py
⚙️ 외부 의존성 (FFmpeg)
MP4 파일 등의 오디오 처리를 위해 FFmpeg가 필요합니다. 다음 두 가지 방법 중 하나를 선택하세요.

자동 인식 (권장): 프로젝트 내 bin/ffmpeg.exe 위치에 실행 파일을 배치합니다.

환경 변수: 시스템 환경 변수 FFMPEG_PATH에 FFmpeg 실행 파일 경로를 지정합니다.

📦 빌드 및 배포 (Build)
PyInstaller를 사용하여 단일 디렉토리(onedir) 방식의 실행 파일을 생성합니다.

PowerShell

.\build_onedir_cpu.bat
빌드 결과물: dist/AI_Mixing_Studio/AI_Mixing_Studio.exe

⚠️ Windows 호환성 및 기술 노트 (Important)
Windows 환경에서의 torchaudio 라이브러리 관련 중요 사항입니다.

문제점: torchaudio 2.9+ 버전부터 오디오 로드/저장(Load/Save) 구현이 TorchCodec 기반으로 변경되었습니다. 현재 TorchCodec은 Windows 환경 지원이 불완전하여, 실행 시 Could not load libtorchcodec 오류가 발생할 수 있습니다.

해결책 (v9 적용): 본 프로젝트는 이 문제를 방지하기 위해 torchaudio.load 함수를 직접 사용하지 않습니다. 대신 numpy 및 soundfile/ffmpeg 기반의 로딩 방식을 사용하여 해당 호환성 문제를 우회합니다.