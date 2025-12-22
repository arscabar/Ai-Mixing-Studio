# src/core/automation.py
import math
from enum import Enum

class InterpolationType(Enum):
    LINEAR = 0
    BEZIER = 1
    STEP = 2

def interpolate(val1: float, val2: float, t: float, mode: InterpolationType) -> float:
    if t <= 0.0: return val1
    if t >= 1.0: return val2

    if mode == InterpolationType.STEP:
        return val1
    elif mode == InterpolationType.BEZIER:
        # Simple Ease-in-out
        t_smooth = t * t * (3 - 2 * t)
        return val1 + (val2 - val1) * t_smooth
    else:
        # Linear
        return val1 + (val2 - val1) * t

def calculate_value_at_time(keyframes: list, current_time: float, default_val: float) -> float:
    if not keyframes:
        return default_val
        
    # 시간순 정렬 가정 (모델에서 정렬됨)
    if current_time <= keyframes[0].time:
        return keyframes[0].value
    if current_time >= keyframes[-1].time:
        return keyframes[-1].value
        
    for i in range(len(keyframes) - 1):
        k1 = keyframes[i]
        k2 = keyframes[i+1]
        
        if k1.time <= current_time <= k2.time:
            duration = k2.time - k1.time
            if duration <= 1e-6: return k1.value
            
            t = (current_time - k1.time) / duration
            return interpolate(k1.value, k2.value, t, k1.interp)
            
    return default_val