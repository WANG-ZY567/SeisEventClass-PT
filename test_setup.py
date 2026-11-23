"""Validate the SeisEventClass-PT environment, EVT6 data and core model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _resolve_gpt2_path() -> Path:
    env_path = os.environ.get("SEISMOLLM_GPT2_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()

    helper = REPO_ROOT / "gpt2_model_path.txt"
    if helper.exists():
        value = helper.read_text(encoding="utf-8").strip()
        if value:
            return Path(value).expanduser()

    return REPO_ROOT / "gpt2_cache" / "models--gpt2" / "snapshots" / "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=os.environ.get("SEISMOLLM_DITING2_DATA", ""),
        help="prepared EVT6 directory (or set SEISMOLLM_DITING2_DATA)",
    )
    parser.add_argument("--skip-data", action="store_true", help="skip the EVT6 file check")
    parser.add_argument("--skip-gpt2", action="store_true", help="skip the local pretrained GPT-2 check")
    parser.add_argument(
        "--smoke-model",
        action="store_true",
        help="run a small random-initialized forward/backward pass without GPT-2 weights",
    )
    return parser.parse_args()


def _check_dependencies():
    print("\n[1/5] Python dependencies")
    modules = {}
    for name in ("torch", "numpy", "pandas", "transformers", "peft", "einops", "h5py", "obspy"):
        try:
            module = __import__(name)
        except ImportError as exc:
            raise RuntimeError(f"missing dependency: {name} ({exc})") from exc
        modules[name] = module
        print(f"[OK] {name}: {getattr(module, '__version__', 'installed')}")
    return modules


def _check_torch(torch) -> None:
    print("\n[2/5] PyTorch device")
    print(f"[OK] PyTorch: {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[OK] GPU memory: {total_gb:.1f} GB")
        if total_gb < 8:
            print("[WARN] Less than 8 GB VRAM: start with --batch-size 1 and disable torch.compile.")
    else:
        print("[WARN] CUDA is unavailable; use --device cpu for diagnostics only.")


def _check_gpt2(transformers, skip: bool) -> None:
    print("\n[3/5] Pretrained GPT-2")
    if skip:
        print("[SKIP] GPT-2 check")
        return

    model_path = _resolve_gpt2_path().resolve()
    if not model_path.exists():
        raise RuntimeError(
            f"GPT-2 snapshot not found: {model_path}\n"
            "Run `python download_gpt2_simple.py` or set SEISMOLLM_GPT2_PATH."
        )
    model = transformers.GPT2Model.from_pretrained(
        str(model_path), output_hidden_states=True, local_files_only=True
    )
    print(f"[OK] Loaded GPT-2 from: {model_path}")
    print(f"[OK] Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    del model


def _check_registry() -> None:
    print("\n[4/5] Project registry")
    from datasets import get_dataset_list
    from models import get_model_list

    datasets = get_dataset_list()
    models = get_model_list()
    if "diting2_evt6" not in datasets:
        raise RuntimeError(f"diting2_evt6 is not registered. Available: {datasets}")
    required_model = "SeisMoLLM_evt6"
    if required_model not in models:
        raise RuntimeError(f"{required_model} is not registered.")
    print(f"[OK] EVT6 dataset registered ({len(datasets)} total names)")
    print(f"[OK] EVT6 model registered ({len(models)} total models)")


def _resolve_wave_path(data_dir: Path, value: object) -> Path:
    path = Path(str(value).strip())
    return path if path.is_absolute() else data_dir / path


def _check_evt6_data(data_arg: str, skip: bool, np, pd) -> None:
    print("\n[5/5] Prepared EVT6 data")
    if skip:
        print("[SKIP] data check")
        return
    if not data_arg:
        raise RuntimeError(
            "No EVT6 directory was supplied. Use --data PATH, set "
            "SEISMOLLM_DITING2_DATA, or pass --skip-data."
        )

    data_dir = Path(data_arg).expanduser().resolve()
    candidates = [
        data_dir / "meta_evt6_train.csv",
        data_dir / "meta_evt6.csv",
        data_dir / "meta_evt6_test.csv",
    ]
    meta_path = next((path for path in candidates if path.exists()), None)
    if meta_path is None:
        raise RuntimeError(f"No meta_evt6*.csv found under: {data_dir}")

    frame = pd.read_csv(meta_path, low_memory=False)
    required = {"key", "part", "_npy_path", "_evt6"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{meta_path.name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError(f"metadata is empty: {meta_path}")
    labels = set(pd.to_numeric(frame["_evt6"], errors="coerce").dropna().astype(int).unique())
    if not labels.issubset(set(range(6))):
        raise RuntimeError(f"invalid EVT6 labels: {sorted(labels)}")

    wave_path = _resolve_wave_path(data_dir, frame.iloc[0]["_npy_path"])
    if not wave_path.exists():
        raise RuntimeError(f"example waveform not found: {wave_path}")
    waveform = np.load(wave_path)
    if waveform.ndim != 2 or waveform.shape[0] != 3:
        raise RuntimeError(f"expected waveform [3,L], got {waveform.shape}: {wave_path}")
    print(f"[OK] Metadata: {meta_path} ({len(frame)} rows, labels={sorted(labels)})")
    print(f"[OK] Waveform: {wave_path} shape={waveform.shape} dtype={waveform.dtype}")


def _smoke_model(torch) -> None:
    print("\n[extra] Synthetic model smoke test")
    from models import create_model

    model = create_model(
        "SeisMoLLM_evt6",
        in_channels=3,
        in_samples=1024,
        llm_start_layer=0,
        llm_layers=1,
        d_model=96,
        llm_pretrain=False,
        llm_freeze=False,
    )
    model.train()
    x = torch.randn(2, 3, 1024)
    y = model(x)
    if tuple(y.shape) != (2, 6):
        raise RuntimeError(f"unexpected model output shape: {tuple(y.shape)}")
    loss = -torch.log(y[:, 0].clamp_min(1e-6)).mean()
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite smoke-test loss: {loss.item()}")
    print(f"[OK] forward/backward output={tuple(y.shape)}, loss={loss.item():.4f}")


def main() -> int:
    args = _args()
    print("=" * 64)
    print("SeisEventClass-PT setup check")
    print("=" * 64)
    try:
        modules = _check_dependencies()
        _check_torch(modules["torch"])
        _check_gpt2(modules["transformers"], args.skip_gpt2)
        _check_registry()
        _check_evt6_data(args.data, args.skip_data, modules["numpy"], modules["pandas"])
        if args.smoke_model:
            _smoke_model(modules["torch"])
    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] {exc}")
        return 1

    print("\n" + "=" * 64)
    print("[OK] Requested checks passed.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
