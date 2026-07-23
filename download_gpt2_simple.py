"""Download GPT-2 from Hugging Face and record the local snapshot path."""

from __future__ import annotations

import os
import subprocess
import sys


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("[INFO] huggingface_hub is missing; installing it now.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])


def main() -> None:
    print("=" * 60)
    print("GPT-2 Download Helper (Hugging Face)")
    print("=" * 60)

    print("\n[1/3] Checking huggingface_hub...")
    ensure_huggingface_hub()
    import huggingface_hub

    print(f"[OK] huggingface_hub version: {huggingface_hub.__version__}")

    print("\n[2/3] Configuring Hugging Face endpoint...")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"[OK] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

    print("\n[3/3] Downloading GPT-2 snapshot...")
    from huggingface_hub import snapshot_download

    try:
        model_path = snapshot_download(
            repo_id="gpt2",
            cache_dir="./gpt2_cache",
            resume_download=True,
            local_files_only=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to download GPT-2: {exc}")
        print("You can also disable pretrained initialization with: --pretrain false")
        sys.exit(1)

    print(f"[OK] GPT-2 snapshot path: {model_path}")

    print("\n[CHECK] Loading GPT-2 locally...")
    try:
        import transformers.models.gpt2 as GPT2

        llm = GPT2.GPT2Model.from_pretrained(
            model_path,
            output_hidden_states=True,
            vocab_size=0,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        print(f"[OK] GPT-2 loaded. Parameters: {sum(p.numel() for p in llm.parameters()) / 1e6:.1f}M")
        del llm
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] GPT-2 was downloaded but failed to load: {exc}")
        sys.exit(1)

    with open("gpt2_model_path.txt", "w", encoding="utf-8") as handle:
        handle.write(model_path)

    print("\n" + "=" * 60)
    print("[OK] Download complete. Path saved to gpt2_model_path.txt")
    print("=" * 60)
    print("Next step: run python test_setup.py or set SEISMOLLM_GPT2_PATH explicitly.")


if __name__ == "__main__":
    main()
