# src/logging_util.py
import logging
from logging.handlers import RotatingFileHandler
import os

def init_logging():
    logger = logging.getLogger("AI_Mixing_Studio")
    logger.setLevel(logging.INFO)
    
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    handler = RotatingFileHandler("logs/app.log", maxBytes=1024*1024, backupCount=3)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger