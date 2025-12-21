# PowerShell build helper (PyInstaller)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
python -m pip install --upgrade pip
pip install pyinstaller
pyinstaller --noconfirm --clean --windowed --name "AI_Mixing_Studio" --add-data "src;src" src\main.py
Write-Host "Build done. See dist\AI_Mixing_Studio"
