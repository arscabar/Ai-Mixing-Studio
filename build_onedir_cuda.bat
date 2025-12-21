@echo off
REM CUDA 토치가 설치된 환경에서 실행하세요.
pyinstaller ^
  --noconsole ^
  --onedir ^
  --name "AI_Mixing_Studio" ^
  --add-data "models_cache;models_cache_seed" ^
  --add-data "bin/ffmpeg.exe;bin" ^
  run.py
