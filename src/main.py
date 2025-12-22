# main.py
import sys
import os
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import StudioMainWindow
from src.cache_bootstrap import ensure_writable_cache, apply_env_for_cache

def main():
    # 1. Cache Init (HuggingFace, Torch)
    cache_path = ensure_writable_cache()
    apply_env_for_cache(cache_path)
    
    # 2. App Start
    app = QApplication(sys.argv)
    
    # 전역 스타일시트
    app.setStyleSheet("QMainWindow { background-color: #1e1e1e; color: white; }")
    
    win = StudioMainWindow()
    win.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()