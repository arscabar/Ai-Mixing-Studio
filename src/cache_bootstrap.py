# src/cache_bootstrap.py
import os
import sys

def ensure_writable_cache():
    # Force HF_HOME to a local folder to avoid permission issues
    cache_root = os.path.join(os.getcwd(), "models_cache")
    if not os.path.exists(cache_root):
        os.makedirs(cache_root)
    return cache_root

def apply_env_for_cache(cache_path):
    os.environ["HF_HOME"] = cache_path
    os.environ["TORCH_HOME"] = cache_path