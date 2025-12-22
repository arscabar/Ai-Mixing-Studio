# 🎛️ AI Mixing Studio

**AI Mixing Studio**는 **딥러닝 기반 음원 분리 및 믹싱 도구**입니다. **GPU 가속(CUDA)** 을 지원하며, **MP4 영상 파일에서 오디오를 자동 추출**해 작업할 수 있습니다.

---

## ✨ 주요 기능 (Key Features)

### ⚡ 스마트 하드웨어 가속

* **NVIDIA GPU 사용 가능 시 자동 활성화 (CUDA)**
* GPU가 없거나 사용 불가능하면 **자동으로 CPU 모드로 전환**

### 🎥 미디어 호환성 (MP4 지원)

* **MP4 입력 시 영상 스트림은 제외**
* **오디오 트랙만 자동 추출**하여 AI 분리 및 레이어 작업에 사용

### 💾 내보내기 (Export)

* 최종 믹싱 결과물을 **고품질 오디오(WAV)** 파일로 저장

### 🚀 배포 최적화 (Bootstrap)

* PyInstaller 빌드 시 **`models_cache_seed` 포함 패키징**
* 사용자 PC 최초 실행 시 모델을 자동 초기화:

  * `models_cache_seed` → `%APPDATA%\AI_Mixing_Studio\models_cache` 로 **자동 복사**

### 📝 로깅 시스템

* 애플리케이션 로그를 **로테이션 방식으로 저장**
* 로그 경로:

  * `%APPDATA%\AI_Mixing_Studio\logs\app.log`

---

## 🛠️ 개발 환경 설정 및 실행 (Development)

### 1) 가상 환경 구성

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2) 의존성 패키지 설치

PC 환경에 맞는 스크립트를 실행하세요.

**CPU 전용 PC**

```powershell
.\install_cpu.bat
```

**NVIDIA GPU (CUDA) PC**

```powershell
.\install_cuda.bat
```

### 3) 모델 다운로드 및 실행

```powershell
python -m src.download_models
python run.py
```

---

## ⚙️ 외부 의존성 (FFmpeg)

MP4 오디오 처리 등을 위해 **FFmpeg**가 필요합니다. 아래 중 하나를 선택하세요.

### ✅ 방법 1) 자동 인식 (권장)

* 프로젝트 내부에 FFmpeg 실행 파일 배치:

  * `bin/ffmpeg.exe`

### ✅ 방법 2) 환경 변수 설정

* 시스템 환경 변수 `FFMPEG_PATH`에 FFmpeg 실행 파일 경로 지정

---

## 📦 빌드 및 배포 (Build)

PyInstaller를 사용하여 **onedir 방식** 실행 파일을 생성합니다.

```powershell
.\build_onedir_cpu.bat
```

빌드 결과물:

* `dist/AI_Mixing_Studio/AI_Mixing_Studio.exe`

---

## ⚠️ 알려진 이슈 (torchaudio 2.9+ / TorchCodec)

### 문제

`torchaudio 2.9+`부터 Load/Save 구현이 **TorchCodec 기반**으로 변경되었습니다.
현재 TorchCodec의 **Windows 지원이 불완전**하여 실행 시 아래 오류가 발생할 수 있습니다.

* `Could not load libtorchcodec`

### 해결책 (v9 적용)

본 프로젝트는 해당 문제를 방지하기 위해 **`torchaudio.load`를 직접 사용하지 않습니다.**
대신 아래 방식으로 우회합니다.

* **numpy + soundfile / ffmpeg 기반 로딩 방식 사용**

---

## 📁 (참고) 주요 경로 정리

* 모델 캐시:

  * `%APPDATA%\AI_Mixing_Studio\models_cache`
* 로그:

  * `%APPDATA%\AI_Mixing_Studio\logs\app.log`
* FFmpeg (권장 위치):

  * `bin/ffmpeg.exe`
