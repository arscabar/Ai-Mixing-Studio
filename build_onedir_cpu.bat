@echo off
REM 1) venv 활성화 후 실행하세요.
REM 2) models_cache 폴더가 있어야(=python -m src.download_models) 포함됩니다.
pyinstaller ^
  --noconsole ^
  --onedir ^
  --name "AI_Mixing_Studio" ^
  --add-data "models_cache;models_cache_seed" ^
  --add-data "bin/ffmpeg.exe;bin" ^
  run.py
