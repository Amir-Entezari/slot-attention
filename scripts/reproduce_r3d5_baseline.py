import argparse
import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.checkpoints import (
    load_checkpoint_model_state,
)
from src.data import (
    SSV2BaselineDataset,
    load_class_subset,
)
from src.evaluation import evaluate
from src.models import (
    build_baseline_model,
    get_baseline_processor_name,
)


MODEL_NAME = "r3d_18"

NUM_CLASSES = 5
NUM_FRAMES = 8

VAL_SIZE = 297

EXPECTED_CORRECT = 191

EXPECTED_TOP1 = (
    EXPECTED_CORRECT
    / VAL_SIZE
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--splits-dir",
        required=True,
    )

    parser.add_argument(
        "--clips-dir",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    # ========================================================
    # Device
    # ========================================================

    if args.device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:
        device = torch.device(
            args.device
        )

    print()
    print("=" * 70)
    print(
        "REPRODUCTION CHECK:"
    )
    print(
        "R3D-18 baseline, 5 classes"
    )
    print("=" * 70)

    print(
        "device:",
        device,
    )

    # ========================================================
    # Dataset metadata
    # ========================================================

    (
        _train_metadata,
        val_metadata,
        _test_metadata,
    ) = load_class_subset(
        NUM_CLASSES,
        splits_dir=args.splits_dir,
    )

    print(
        "validation samples:",
        len(val_metadata),
    )

    assert (
        len(val_metadata)
        == VAL_SIZE
    ), (
        f"Expected {VAL_SIZE} validation "
        f"samples, got {len(val_metadata)}."
    )

    # ========================================================
    # Processor
    #
    # Historical R3D experiments used the ViT image processor.
    # We preserve that behavior for canonical reproduction.
    # ========================================================

    processor_name = (
        get_baseline_processor_name(
            MODEL_NAME
        )
    )

    expected_processor = (
        "google/vit-base-patch16-224"
    )

    print(
        "processor:",
        processor_name,
    )

    assert (
        processor_name
        == expected_processor
    )

    # ========================================================
    # Validation loader
    # ========================================================

    val_dataset = SSV2BaselineDataset(
        val_metadata,
        args.clips_dir,
        num_frames=NUM_FRAMES,
        processor_name=processor_name,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # ========================================================
    # Model
    # ========================================================

    model = build_baseline_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        num_frames=NUM_FRAMES,
    ).to(
        device
    )

    # ========================================================
    # Checkpoint
    # ========================================================

    if not os.path.exists(
        args.checkpoint
    ):
        raise FileNotFoundError(
            args.checkpoint
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    print()
    print(
        "checkpoint:",
        args.checkpoint,
    )

    print(
        "checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "checkpoint best epoch:",
        checkpoint.get(
            "best_epoch",
            "unknown",
        ),
    )

    print(
        "checkpoint best val:",
        checkpoint.get(
            "best_val",
            "unknown",
        ),
    )

    # ========================================================
    # Metadata checks when present
    # ========================================================

    expected_metadata = {
        "model_name":
            MODEL_NAME,

        "num_classes":
            NUM_CLASSES,

        "num_frames":
            NUM_FRAMES,
    }

    for (
        key,
        expected_value,
    ) in expected_metadata.items():

        if key in checkpoint:

            actual_value = (
                checkpoint[key]
            )

            assert (
                actual_value
                == expected_value
            ), (
                f"Checkpoint {key} mismatch: "
                f"{actual_value!r} != "
                f"{expected_value!r}"
            )

    # ========================================================
    # Restore checkpoint
    #
    # R3D is different from ViT/TimeSformer/VideoMAE:
    # the whole backbone is trainable, so there should be
    # no intentionally missing keys.
    # ========================================================

    missing = (
        load_checkpoint_model_state(
            model,
            checkpoint[
                "model_state_dict"
            ],
        )
    )

    print(
        "missing checkpoint keys:",
        len(missing),
    )

    if missing:
        print(
            "missing:",
            missing,
        )

    assert not missing, (
        "R3D checkpoint should contain the "
        "entire model state, but some keys "
        f"were missing: {missing}"
    )

    # ========================================================
    # Optional structural sanity check
    # ========================================================

    backbone_state_keys = [
        key
        for key
        in checkpoint[
            "model_state_dict"
        ]
        if key.startswith(
            "backbone."
        )
    ]

    print(
        "stored backbone keys:",
        len(backbone_state_keys),
    )

    assert backbone_state_keys, (
        "R3D checkpoint does not appear "
        "to contain backbone parameters."
    )

    # ========================================================
    # Evaluation
    # ========================================================

    criterion = (
        nn.CrossEntropyLoss()
    )

    (
        val_loss,
        val_metrics,
    ) = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        topk_values=[1],
        device=device,
    )

    top1 = (
        val_metrics["top1"]
    )

    correct = round(
        top1
        * VAL_SIZE
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"validation loss: "
        f"{val_loss:.10f}"
    )

    print(
        f"Top-1: "
        f"{top1:.10f}"
    )

    print(
        "correct:",
        correct,
        "/",
        VAL_SIZE,
    )

    print()

    print(
        "canonical Top-1:",
        f"{EXPECTED_TOP1:.10f}",
    )

    print(
        "canonical correct:",
        EXPECTED_CORRECT,
        "/",
        VAL_SIZE,
    )

    print(
        "Top-1 difference:",
        abs(
            top1
            - EXPECTED_TOP1
        ),
    )

    # ========================================================
    # Hard reproduction check
    # ========================================================

    assert (
        correct
        == EXPECTED_CORRECT
    ), (
        "FAILED: R3D baseline does not "
        "reproduce the canonical number "
        "of correct validation predictions."
    )

    assert abs(
        top1
        - EXPECTED_TOP1
    ) < 1e-12

    print()
    print(
        "✓ CANONICAL R3D-18 BASELINE "
        "REPRODUCTION PASSED"
    )


if __name__ == "__main__":
    main()