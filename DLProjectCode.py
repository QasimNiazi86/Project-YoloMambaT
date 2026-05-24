#!/usr/bin/env python
"""Local GPU runner for Mamba-YOLO.

This replaces the original Google Colab notebook export. It assumes the
Mamba-YOLO repository lives next to this file and writes training outputs to a
local folder instead of Google Drive.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_DIR = ROOT / "Mamba-YOLO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mamba-YOLO locally on GPU.")
    parser.add_argument("--task", choices=("train", "val"), default="train")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/VOC.yaml")
    parser.add_argument("--config", default="ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml")
    parser.add_argument("--project", default=str(ROOT / "teacher_baseline"))
    parser.add_argument("--name", default="mamba_yolo_t_voc")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device id, for example 0, 1, 0,1, or cpu.")
    parser.add_argument("--optimizer", default="SGD", choices=("SGD", "Adam", "AdamW"))
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--resume", default=None, help="Path to a local checkpoint, e.g. teacher_baseline/.../last.pt")
    parser.add_argument("--dry-run", action="store_true", help="Check imports, CUDA, and paths without training.")
    return parser.parse_args()


def require_repo() -> None:
    if not REPO_DIR.exists():
        raise FileNotFoundError(
            f"Expected Mamba-YOLO repo at {REPO_DIR}. Clone it there before running this script."
        )


def configure_imports() -> None:
    os.chdir(REPO_DIR)
    sys.path.insert(0, str(REPO_DIR))


def check_gpu(device: str) -> None:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if device.lower() == "cpu":
        print("Running on CPU because --device cpu was requested.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. Install a CUDA-enabled PyTorch build "
            "or run with --device cpu."
        )

    visible_devices = torch.cuda.device_count()
    requested = [int(part) for part in device.split(",") if part.strip().isdigit()]
    bad = [idx for idx in requested if idx >= visible_devices]
    if bad:
        raise RuntimeError(f"Requested CUDA device(s) {bad}, but only {visible_devices} device(s) are visible.")

    print(f"CUDA available: {torch.cuda.get_device_name(requested[0] if requested else 0)}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA toolkit: {CUDA_HOME or 'not found'}")


def check_selective_scan_extension() -> None:
    try:
        import selective_scan_cuda_core  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Mamba-YOLO's selective_scan CUDA extension is not installed. Build it after "
            "installing a PyTorch CUDA wheel that matches your local CUDA toolkit. For this "
            "machine, PyTorch is currently built for CUDA 11.8 while the installed toolkit is "
            "CUDA 12.8, so `pip install ./Mamba-YOLO/selective_scan` cannot compile yet."
        ) from exc


def patch_torch_load_for_old_ultralytics() -> None:
    """Allow older Ultralytics checkpoints to load under newer PyTorch defaults."""
    import torch

    if hasattr(torch.load, "_mamba_yolo_local_patch"):
        return

    original_torch_load = torch.load

    def unsafe_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    unsafe_load._mamba_yolo_local_patch = True
    torch.load = unsafe_load


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((REPO_DIR / candidate).resolve())


def prepare_data_yaml(path: str) -> str:
    data_path = Path(resolve_path(path))
    if data_path.name.lower() != "voc.yaml":
        return str(data_path)

    local_dataset = ROOT / "datasets" / "VOC"
    local_config_dir = ROOT / "local_configs"
    local_config_dir.mkdir(exist_ok=True)
    local_data_yaml = local_config_dir / "VOC.local.yaml"

    text = data_path.read_text(encoding="utf-8")
    local_path = local_dataset.as_posix()
    text = re.sub(r"(?m)^path:\s*.*$", f"path: {local_path}", text, count=1)
    local_data_yaml.write_text(text, encoding="utf-8")
    return str(local_data_yaml)


def main() -> None:
    args = parse_args()
    require_repo()
    configure_imports()

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("YOLO_VERBOSE", "True")

    check_gpu(args.device)
    check_selective_scan_extension()
    patch_torch_load_for_old_ultralytics()

    from ultralytics import YOLO

    data = prepare_data_yaml(args.data)
    project = str(Path(args.project).resolve())

    if args.dry_run:
        print(f"Repository: {REPO_DIR}")
        print(f"Data YAML: {data}")
        print(f"Config YAML: {resolve_path(args.config)}")
        print(f"Output folder: {project}")
        print("Dry run passed.")
        return

    if args.resume:
        model = YOLO(str(Path(args.resume).resolve()))
        model.train(resume=True, device=args.device)
        return

    model = YOLO(resolve_path(args.config))
    run_args = {
        "data": data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "device": args.device,
        "amp": args.amp,
        "project": project,
        "name": args.name,
    }

    if args.task == "train":
        model.train(**run_args)
    else:
        model.val(**run_args)


if __name__ == "__main__":
    main()
