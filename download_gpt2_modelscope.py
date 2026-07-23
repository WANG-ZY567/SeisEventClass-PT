"""Download GPT-2 from ModelScope and record the local snapshot path."""

from __future__ import annotations

import subprocess
import sys


def ensure_modelscope() -> None:
    try:
        import modelscope  # noqa: F401
    except ImportError:
        print("[INFO] modelscope is missing; installing it now.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])


def main() -> None:
    print("=" * 60)
    print("GPT-2 Download Helper (ModelScope)")
    print("=" * 60)

    print("\n[1/3] Checking modelscope...")
    ensure_modelscope()
    import modelscope

    print(f"[OK] modelscope version: {modelscope.__version__}")

    print("\n[2/3] Downloading GPT-2 snapshot...")
    from modelscope import snapshot_download

    try:
        model_dir = snapshot_download(
            "AI-ModelScope/gpt2",
            cache_dir="./pretrained_models",
            revision="master",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to download GPT-2: {exc}")
        sys.exit(1)

    print(f"[OK] GPT-2 snapshot path: {model_dir}")

    print("\n[3/3] Loading GPT-2 locally...")
    try:
        import transformers.models.gpt2 as GPT2

        llm = GPT2.GPT2Model.from_pretrained(
            model_dir,
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
        handle.write(model_dir)

    print("\n" + "=" * 60)
    print("[OK] Download complete. Path saved to gpt2_model_path.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
