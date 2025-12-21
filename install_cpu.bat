@echo off
REM ✅ Windows에서는 torchaudio 2.9+가 torchcodec 기반으로 변경되어 오류가 날 수 있어
REM    torchaudio 2.8.* 조합을 권장합니다.
python -m pip install -U pip
pip install -r requirements_base.txt
pip uninstall -y torch torchvision torchaudio torchcodec
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
