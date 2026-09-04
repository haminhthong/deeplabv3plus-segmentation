from __future__ import annotations

from pathlib import Path

import pytest
import torch

from inference import build_model, load_checkpoint_model


def test_save_and_load_checkpoint(tmp_path: Path):
    ckpt_path = tmp_path / "test_model.pth"
    model = build_model("resnet50", None, 21, "deeplabv3plus")

    metadata = {
        "epoch": 10,
        "model_state_dict": model.state_dict(),
        "architecture": "deeplabv3plus",
        "encoder": "resnet50",
        "image_size": 320,
        "num_classes": 21,
        "best_val_miou": 0.75,
        "git_commit": "abcdef123456",
    }
    torch.save(metadata, ckpt_path)

    device = torch.device("cpu")
    loaded_model, loaded_meta = load_checkpoint_model(ckpt_path, device)

    assert loaded_model is not None
    assert loaded_meta["architecture"] == "deeplabv3plus"
    assert loaded_meta["encoder"] == "resnet50"
    assert loaded_meta["best_val_miou"] == 0.75
    assert loaded_meta["git_commit"] == "abcdef123456"


def test_load_checkpoint_missing_file(tmp_path: Path):
    device = torch.device("cpu")
    with pytest.raises(FileNotFoundError):
        load_checkpoint_model(tmp_path / "missing.pth", device)
