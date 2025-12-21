# src/audio_io.py
import numpy as np

def ensure_stereo_np(audio_data: np.ndarray) -> np.ndarray:
    if audio_data.ndim == 1:
        # Mono -> Stereo
        return np.stack([audio_data, audio_data], axis=1)
    elif audio_data.ndim == 2:
        if audio_data.shape[1] == 1:
            return np.concatenate([audio_data, audio_data], axis=1)
        elif audio_data.shape[1] > 2:
            return audio_data[:, :2] # Crop extra channels
    return audio_data