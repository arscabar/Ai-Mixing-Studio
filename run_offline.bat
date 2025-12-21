@echo off
cd /d %~dp0
call .venv\Scripts\activate

set HF_HOME=%CD%\models_cache
set TRANSFORMERS_OFFLINE=1
set HF_HUB_OFFLINE=1
set TORCH_HOME=%CD%\models_cache

python -m src.main
