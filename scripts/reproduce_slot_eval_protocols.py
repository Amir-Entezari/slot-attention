import argparse
import gc
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
    evaluate,
    evaluate_slot_per_video_init,
)
from src.models import (
    build_slot_model,
    get_slot_processor_name,
)


MODEL_NAME = "videomae_controlled"

NUM_CLASSES = 5
NUM_FRAMES = 16

NUM_SLOTS = 4
SLOT_DIM = 128
SLOT_ITERS = 3

VAL_SIZE = 297

LEGACY_SEEDS = [
    0,
    1,
    2,
    3,
    4,
    42,
]

EXPECTED_LEGACY_CORRECT = {
    0: 213,
    1: 213,
    2: 214,
    3: 210,
    4: 222,
    42: 219,
}

PER_VIDEO_SEED = 12345
EXPECTED_CORRECTED = 214


def build_and_load(
    checkpoint_state,
    eval_seed,
    device,
):
    model = build_slot_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        num_frames=NUM_FRAMES,
        num_slots=NUM_SLOTS,
        slot_dim=SLOT_DIM,
        slot_iters=SLOT_ITERS,
        eval_seed=eval_seed,
    ).to(device)

    load_checkpoint_model_state(
        model,
        checkpoint_state,
    )

    return model


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
        default=2,
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
    print("=" * 72)
    print(
        "SLOT EVALUATION PROTOCOL REPRODUCTION"
    )
    print("=" * 72)

    print(
        "device:",
        device,
    )

    # ========================================================
    # Dataset
    # ========================================================

    (
        _train_metadata,
        val_metadata,
        _test_metadata,
    ) = load_class_subset(
        NUM_CLASSES,
        splits_dir=args.splits_dir,
    )

    assert len(
        val_metadata
    ) == VAL_SIZE, (
        f"Expected {VAL_SIZE} validation "
        f"examples, got {len(val_metadata)}."
    )

    processor_name = (
        get_slot_processor_name(
            MODEL_NAME,
            slot_eval_mode=(
                "per_video_deterministic"
            ),
        )
    )

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

    print(
        "validation samples:",
        len(val_dataset),
    )

    print(
        "processor:",
        processor_name,
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

    checkpoint_state = (
        checkpoint[
            "model_state_dict"
        ]
    )

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

    # ========================================================
    # Criterion
    # ========================================================

    criterion = (
        nn.CrossEntropyLoss()
    )

    # ========================================================
    # 1. Historical shared-initialization evaluation
    # ========================================================

    print()
    print("=" * 72)
    print(
        "LEGACY SHARED INITIALIZATION"
    )
    print("=" * 72)

    legacy_results = {}

    for eval_seed in LEGACY_SEEDS:

        model = build_and_load(
            checkpoint_state=(
                checkpoint_state
            ),
            eval_seed=eval_seed,
            device=device,
        )

        (
            loss,
            metrics,
        ) = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            topk_values=[1],
            device=device,
        )

        top1 = metrics["top1"]

        correct = round(
            top1 * VAL_SIZE
        )

        legacy_results[
            eval_seed
        ] = {
            "loss":
                loss,

            "top1":
                top1,

            "correct":
                correct,
        }

        expected_correct = (
            EXPECTED_LEGACY_CORRECT[
                eval_seed
            ]
        )

        print(
            f"seed={eval_seed:>2} "
            f"loss={loss:.6f} "
            f"top1={top1:.10f} "
            f"correct="
            f"{correct}/{VAL_SIZE} "
            f"expected="
            f"{expected_correct}/{VAL_SIZE}"
        )

        assert (
            correct
            == expected_correct
        ), (
            "Legacy shared-init reproduction "
            f"failed for eval_seed={eval_seed}: "
            f"got {correct}, "
            f"expected {expected_correct}."
        )

        del model

        gc.collect()

        if (
            device.type
            == "cuda"
        ):
            torch.cuda.empty_cache()

    # ========================================================
    # Legacy mean
    # ========================================================

    legacy_mean = sum(
        result["top1"]
        for result
        in legacy_results.values()
    ) / len(
        legacy_results
    )

    expected_legacy_mean = (
        sum(
            EXPECTED_LEGACY_CORRECT.values()
        )
        / (
            len(
                EXPECTED_LEGACY_CORRECT
            )
            * VAL_SIZE
        )
    )

    print()
    print(
        "legacy mean Top-1:",
        f"{legacy_mean:.10f}",
    )

    print(
        "expected mean:",
        f"{expected_legacy_mean:.10f}",
    )

    assert abs(
        legacy_mean
        - expected_legacy_mean
    ) < 1e-12

    # ========================================================
    # 2. Corrected per-video deterministic evaluation
    # ========================================================

    print()
    print("=" * 72)
    print(
        "CORRECTED PER-VIDEO INITIALIZATION"
    )
    print("=" * 72)

    # eval_seed is irrelevant here because explicit
    # initial_slots are supplied by the corrected evaluator.
    model = build_and_load(
        checkpoint_state=(
            checkpoint_state
        ),
        eval_seed=0,
        device=device,
    )

    (
        corrected_loss,
        corrected_metrics,
    ) = evaluate_slot_per_video_init(
        model=model,
        loader=val_loader,
        criterion=criterion,
        topk_values=[1],
        device=device,
        init_seed=PER_VIDEO_SEED,
    )

    corrected_top1 = (
        corrected_metrics[
            "top1"
        ]
    )

    corrected_correct = round(
        corrected_top1
        * VAL_SIZE
    )

    print(
        f"init_seed="
        f"{PER_VIDEO_SEED} "
        f"loss="
        f"{corrected_loss:.6f} "
        f"top1="
        f"{corrected_top1:.10f} "
        f"correct="
        f"{corrected_correct}/"
        f"{VAL_SIZE}"
    )

    assert (
        corrected_correct
        == EXPECTED_CORRECTED
    ), (
        "Corrected per-video evaluation "
        "did not reproduce the canonical "
        f"{EXPECTED_CORRECTED}/{VAL_SIZE}."
    )

    expected_corrected_top1 = (
        EXPECTED_CORRECTED
        / VAL_SIZE
    )

    assert abs(
        corrected_top1
        - expected_corrected_top1
    ) < 1e-12

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for seed in LEGACY_SEEDS:
        result = (
            legacy_results[
                seed
            ]
        )

        print(
            f"legacy seed "
            f"{seed:>2}: "
            f"{result['correct']:>3}/"
            f"{VAL_SIZE} "
            f"= "
            f"{result['top1']:.4f}"
        )

    print(
        "legacy mean:    "
        f"{legacy_mean:.4f}"
    )

    print(
        "corrected:      "
        f"{corrected_correct}/"
        f"{VAL_SIZE} "
        f"= "
        f"{corrected_top1:.4f}"
    )

    print()
    print(
        "✓ LEGACY AND CORRECTED "
        "SLOT EVALUATION PROTOCOLS "
        "REPRODUCED"
    )


if __name__ == "__main__":
    main()