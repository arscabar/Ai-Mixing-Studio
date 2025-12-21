"""온라인 1회 실행하여 모델을 models_cache에 받아두는 스크립트"""
import os

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    cache_dir = os.path.join(root, "models_cache")
    os.makedirs(cache_dir, exist_ok=True)

    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["TORCH_HOME"] = cache_dir

    print("Downloading Transformers AST model...")
    from transformers import pipeline
    _ = pipeline("audio-classification", model="mit/ast-finetuned-audioset-10-10-0.4593", device=-1)

    print("Downloading TorchAudio HDemucs bundle...")
    import torchaudio
    _ = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.get_model()

    print("Done.")
    print("Cache folder:", cache_dir)

if __name__ == "__main__":
    main()
