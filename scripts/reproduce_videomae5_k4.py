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
from src.evaluation import (
    evaluate_slot_per_video_init,
)
from src.models import (
    build_slot_model,
    get_slot_processor_name,
)


# ============================================================
# Canonical reference configuration
# ============================================================

MODEL_NAME = "videomae_controlled"

NUM_CLASSES = 5
NUM_FRAMES = 16

NUM_SLOTS = 4
SLOT_DIM = 128
SLOT_ITERS = 3

TRAINING_SEED = 42
EVAL_INIT_SEED = 12345

BATCH_SIZE = 2
NUM_WORKERS = 2

EXPECTED_ORIGINAL_LABELS = [
    111,
    44,
    93,
    42,
    94,
]

EXPECTED_VAL_SIZE = 297

EXPECTED_CORRECT = 214

EXPECTED_TOP1 = (
    EXPECTED_CORRECT
    / EXPECTED_VAL_SIZE
)

EXPECTED_VAL_LOSS_APPROX = 0.7320


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
        "VideoMAE + Slot, 5 classes, K=4"
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

    print()
    print(
        "validation samples:",
        len(val_metadata),
    )

    assert (
        len(val_metadata)
        == EXPECTED_VAL_SIZE
    ), (
        "Validation-set size mismatch: "
        f"got {len(val_metadata)}, "
        f"expected {EXPECTED_VAL_SIZE}"
    )

    # ========================================================
    # Processor
    # ========================================================

    processor_name = (
        get_slot_processor_name(
            MODEL_NAME,
            slot_eval_mode=(
                "per_video_deterministic"
            ),
        )
    )

    expected_processor = (
        "MCG-NJU/"
        "videomae-base-finetuned-kinetics"
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
    # Validation dataset
    #
    # Deliberately mirrors the canonical notebook directly.
    # ========================================================

    val_dataset = SSV2BaselineDataset(
        val_metadata,
        args.clips_dir,
        num_frames=NUM_FRAMES,
        processor_name=processor_name,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # ========================================================
    # Model
    # ========================================================

    model = build_slot_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        num_frames=NUM_FRAMES,
        num_slots=NUM_SLOTS,
        slot_dim=SLOT_DIM,
        slot_iters=SLOT_ITERS,

        # The canonical corrected run was built with
        # training seed 42 here.
        #
        # Explicit initial_slots are used during corrected
        # evaluation, so _eval_noise is bypassed, but we still
        # reproduce the notebook configuration exactly.
        eval_seed=TRAINING_SEED,
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

    # --------------------------------------------------------
    # Check canonical metadata when present.
    # Older checkpoints may omit some fields.
    # --------------------------------------------------------

    expected_metadata = {
        "model_name":
            MODEL_NAME,

        "num_classes":
            NUM_CLASSES,

        "num_frames":
            NUM_FRAMES,

        "num_slots":
            NUM_SLOTS,

        "slot_dim":
            SLOT_DIM,

        "slot_iters":
            SLOT_ITERS,

        "seed":
            TRAINING_SEED,
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

    # The canonical notebook used this older metadata name.
    if (
        "slot_eval_init_seed"
        in checkpoint
    ):
        assert (
            checkpoint[
                "slot_eval_init_seed"
            ]
            == EVAL_INIT_SEED
        )

    load_checkpoint_model_state(
        model,
        checkpoint[
            "model_state_dict"
        ],
    )

    # ========================================================
    # Corrected evaluation
    # ========================================================

    criterion = (
        nn.CrossEntropyLoss()
    )

    (
        val_loss,
        val_metrics,
    ) = evaluate_slot_per_video_init(
        model=model,
        loader=val_loader,
        criterion=criterion,
        topk_values=[1],
        device=device,
        init_seed=EVAL_INIT_SEED,
    )

    actual_top1 = (
        val_metrics["top1"]
    )

    actual_correct = round(
        actual_top1
        * EXPECTED_VAL_SIZE
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
        f"{actual_top1:.10f}"
    )

    print(
        "correct predictions:",
        actual_correct,
        "/",
        EXPECTED_VAL_SIZE,
    )

    print()
    print(
        "canonical loss:",
        f"~{EXPECTED_VAL_LOSS_APPROX:.4f}",
    )

    print(
        "canonical Top-1:",
        f"{EXPECTED_TOP1:.10f}",
    )

    print(
        "canonical correct:",
        EXPECTED_CORRECT,
        "/",
        EXPECTED_VAL_SIZE,
    )

    print()
    print(
        "Top-1 difference:",
        abs(
            actual_top1
            - EXPECTED_TOP1
        ),
    )

    print(
        "loss difference from "
        "rounded notebook value:",
        abs(
            val_loss
            - EXPECTED_VAL_LOSS_APPROX
        ),
    )

    # ========================================================
    # Hard reproduction checks
    # ========================================================

    assert (
        actual_correct
        == EXPECTED_CORRECT
    ), (
        "FAILED: clean repo does not reproduce "
        "the canonical number of correct "
        "validation predictions."
    )

    assert abs(
        actual_top1
        - EXPECTED_TOP1
    ) < 1e-12

    # The notebook only printed loss to four decimal places,
    # and GPU/library differences can cause tiny numerical
    # variation. Use a small tolerance rather than requiring
    # bitwise equality.
    assert abs(
        val_loss
        - EXPECTED_VAL_LOSS_APPROX
    ) < 5e-4, (
        "Top-1 reproduced, but validation loss "
        "is farther from the canonical 0.7320 "
        "than expected."
    )

    print()
    print(
        "✓ CANONICAL VIDEOMAE K=4 "
        "REPRODUCTION PASSED"
    )


if __name__ == "__main__":
    main()