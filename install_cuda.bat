@echo off
REM ✅ CUDA 환경 (본인 CUDA에 맞는 cuXXX로 변경)
python -m pip install -U pip
pip install -r requirements_base.txt
pip uninstall -y torch torchvision torchaudio torchcodec
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
