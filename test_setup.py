"""Validate the local SeisEventClass-PT environment and data paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def _resolve_gpt2_path() -> str:
    """Resolve the GPT-2 path from environment variables, a helper file, or the default cache."""
    env_path = os.environ.get("SEISMOLLM_GPT2_PATH", "").strip()
    if env_path:
        return env_path

    txt = Path(__file__).parent / "gpt2_model_path.txt"
    if txt.exists():
        path = txt.read_text(encoding="utf-8").strip().replace("\\", "/")
        if path:
            return path

    return "./gpt2_cache/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"


def main() -> None:
    print("=" * 60)
    print("SeisEventClass-PT Setup Check")
    print("=" * 60)

    print("\n[1/5] Checking Python dependencies...")
    try:
        import numpy as np
        import pandas as pd
        import transformers

        try:
            import peft

            print(f"[OK] peft: {peft.__version__}")
        except ImportError:
            print("[WARN] peft is not installed. LoRA-based fine-tuning requires peft.")

        print(f"[OK] transformers: {transformers.__version__}")
        print(f"[OK] pandas: {pd.__version__}")
        print(f"[OK] numpy: {np.__version__}")
    except ImportError as exc:
        print(f"[ERROR] Missing dependency: {exc}")
        sys.exit(1)

    print("\n[2/5] Checking PyTorch and CUDA...")
    print(f"[OK] PyTorch: {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[OK] GPU count: {torch.cuda.device_count()}")
    else:
        print("[WARN] CUDA is not available. Training will run on CPU if attempted.")

    print("\n[3/5] Checking GPT-2 snapshot...")
    gpt2_model_path = _resolve_gpt2_path()
    if not os.path.exists(gpt2_model_path):
        print(f"[ERROR] GPT-2 path does not exist: {gpt2_model_path}")
        print("Run: python download_gpt2_simple.py")
        print("Or set: export SEISMOLLM_GPT2_PATH=/abs/path/to/gpt2/snapshot")
        sys.exit(1)

    try:
        import transformers.models.gpt2 as GPT2

        print("[INFO] Loading GPT-2...")
        llm = GPT2.GPT2Model.from_pretrained(
            gpt2_model_path,
            output_hidden_states=True,
            vocab_size=0,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        print(f"[OK] GPT-2 loaded. Parameters: {sum(p.numel() for p in llm.parameters()) / 1e6:.1f}M")
        del llm
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load GPT-2: {exc}")
        sys.exit(1)

    print("\n[4/5] Checking dataset registry...")
    try:
        from datasets import get_dataset_list

        dataset_list = get_dataset_list()
        print(f"[OK] Registered datasets: {dataset_list}")
        if any("diting2" in ds.lower() for ds in dataset_list):
            print("[OK] DiTing2 dataset loaders are registered.")
        else:
            print("[ERROR] DiTing2 dataset loaders were not found in the registry.")
            sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Dataset registry check failed: {exc}")
        sys.exit(1)

    print("\n[5/5] Checking prepared DiTing2 data...")
    default_data_path = "/path/to/diting2_preprocessed"
    data_path = os.environ.get("SEISMOLLM_DITING2_DATA", default_data_path)
    meta_csv = os.path.join(data_path, "meta_full5.csv")

    if not os.path.exists(meta_csv):
        print(f"[ERROR] Metametadata CSV not found: {meta_csv}")
        print("Prepare the data with tools/prepare_diting2.py, then set:")
        print("  export SEISMOLLM_DITING2_DATA=/abs/path/to/diting2_preprocessed")
        sys.exit(1)

    try:
        import numpy as np
        import pandas as pd

        df = pd.read_csv(meta_csv)
        print(f"[OK] Metadata rows: {len(df)}")
        print(f"[OK] Metadata columns: {list(df.columns)}")

        print("\nColumn dtypes:")
        for col in ["key", "_pmp_bin", "baz", "dis", "evmag"]:
            if col in df.columns:
                print(f"  {col}: {df[col].dtype}")

        waves_dir = os.path.join(data_path, "waves")
        if "_npy_path" in df.columns:
            npy_path = os.path.join(data_path, df.iloc[0]["_npy_path"])
        else:
            first_key = df.iloc[0]["key"]
            first_part = df.iloc[0]["part"]
            npy_path = os.path.join(waves_dir, f"{first_key}_{first_part}.npy")

        if not os.path.exists(npy_path):
            print(f"[ERROR] Example waveform file not found: {npy_path}")
            sys.exit(1)

        waveform = np.load(npy_path)
        print("[OK] Example waveform loaded.")
        print(f"  shape: {waveform.shape}")
        print(f"  dtype: {waveform.dtype}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Data check failed: {exc}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("[OK] Environment and data checks passed.")
    print("=" * 60)
    print("Next step: run the training commands described in README.md and docs/REPRODUCIBILITY.md.")


if __name__ == "__main__":
    main()
