import argparse
import math

import torch

from src.checkpoints import (
    load_checkpoint_model_state,
)

from src.data import (
    build_dataloaders,
    build_ssv2_datasets,
    load_class_subset,
)

from src.diagnostics import (
    _build_per_video_initial_slots,
    collect_slot_diagnostics,
    summarize_diagnostic_rows,
    videomae_temporal_statistics,
)

from src.models import (
    build_slot_model,
    get_slot_processor_name,
)


# ============================================================
# Canonical configuration
# ============================================================

MODEL_NAME = "videomae_controlled"

NUM_CLASSES = 5
NUM_FRAMES = 16

NUM_SLOTS = 4
SLOT_DIM = 128
SLOT_ITERS = 3

TRAINING_SEED = 42
SLOT_EVAL_SEED = 12345

EXPECTED_VAL_SIZE = 297
EXPECTED_CORRECT = 214
EXPECTED_TOP1 = (
    EXPECTED_CORRECT
    / EXPECTED_VAL_SIZE
)


# ============================================================
# Retained canonical corrected-diagnostic means
#
# These are recorded to four decimal places.
#
# Therefore reproduction uses a tolerance rather than
# pretending these are full-precision reference values.
# ============================================================

EXPECTED_DIAGNOSTICS = {
    "slot_cos_mean":
        0.6954,

    "attn_cos_mean":
        0.4077,

    "attn_js_mean":
        0.5201,

    "assign_entropy_mean":
        0.1457,

    "assignment_confidence":
        0.9178,

    "effective_slots":
        1.4588,

    "soft_usage_max":
        0.8997,

    "hard_usage_max":
        0.9611,

    "hard_dead_slots":
        0.7441,

    "slot_token_entropy_mean":
        0.7313,
}


# Four-decimal historical values + minor numerical variation.
DIAGNOSTIC_TOLERANCE = 0.005


def check_real_temporal_integration(
    model,
    val_loader,
    device,
):
    """
    Run the new temporal/tubelet diagnostic machinery on a
    real VideoMAE batch.

    This is a structural integration check, not comparison
    against an old canonical temporal metric, because the
    temporal diagnostics are newly migrated/expanded.
    """

    videos, _labels = next(
        iter(val_loader)
    )

    videos = videos.to(
        device
    )

    batch_size = (
        videos.shape[0]
    )

    # --------------------------------------------------------
    # Reproduce the first B entries of the same deterministic
    # per-video initialization bank used in corrected eval.
    # --------------------------------------------------------

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        SLOT_EVAL_SEED
    )

    noise = torch.randn(
        batch_size,
        NUM_SLOTS,
        SLOT_DIM,
        generator=generator,
    )

    initial_slots = (
        _build_per_video_initial_slots(
            model,
            noise,
        )
    )

    # --------------------------------------------------------
    # Real VideoMAE diagnostic forward
    # --------------------------------------------------------

    with torch.inference_mode():

        tokens = (
            model
            .extract_projected_tokens(
                videos
            )
        )

        print()
        print(
            "real projected token shape:",
            tuple(tokens.shape),
        )

        assert (
            tokens.ndim
            == 3
        )

        assert (
            tokens.shape[1]
            == 1568
        ), (
            "Expected VideoMAE token geometry "
            "8*14*14 = 1568."
        )

        (
            _slots,
            attn,
            assignment,
        ) = model.slot_attn(
            tokens,
            initial_slots=initial_slots,
            return_assignment=True,
        )

    temporal = (
        videomae_temporal_statistics(
            assignment,
            attn,
            temporal_bins=8,
            grid_height=14,
            grid_width=14,
        )
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    assert tuple(
        temporal[
            "assignment_grid"
        ].shape
    ) == (
        batch_size,
        NUM_SLOTS,
        8,
        14,
        14,
    )

    assert tuple(
        temporal[
            "attn_temporal_profile"
        ].shape
    ) == (
        batch_size,
        NUM_SLOTS,
        8,
    )

    assert tuple(
        temporal[
            "assignment_temporal_usage"
        ].shape
    ) == (
        batch_size,
        NUM_SLOTS,
        8,
    )

    # Each slot's temporal attention profile must sum to 1.
    assert torch.allclose(
        temporal[
            "attn_temporal_profile"
        ].sum(dim=-1),
        torch.ones(
            batch_size,
            NUM_SLOTS,
            device=(
                temporal[
                    "attn_temporal_profile"
                ].device
            ),
            dtype=(
                temporal[
                    "attn_temporal_profile"
                ].dtype
            ),
        ),
        atol=1e-4,
        rtol=1e-4,
    )

    # At every tubelet, assignment usage over slots sums to 1.
    assert torch.allclose(
        temporal[
            "assignment_temporal_usage"
        ].sum(dim=1),
        torch.ones(
            batch_size,
            8,
            device=(
                temporal[
                    "assignment_temporal_usage"
                ].device
            ),
            dtype=(
                temporal[
                    "assignment_temporal_usage"
                ].dtype
            ),
        ),
        atol=1e-4,
        rtol=1e-4,
    )

    print(
        "temporal entropy mean:",
        float(
            temporal[
                "temporal_entropy_mean"
            ].mean().item()
        ),
    )

    print(
        "temporal cosine mean:",
        float(
            temporal[
                "temporal_cos_mean"
            ].mean().item()
        ),
    )

    print(
        "temporal JS mean:",
        float(
            temporal[
                "temporal_js_mean"
            ].mean().item()
        ),
    )

    print(
        "✓ REAL VIDEOMAE TEMPORAL "
        "DIAGNOSTIC INTEGRATION PASSED"
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
    print("=" * 74)
    print(
        "VIDEOMAE K=4 CORRECTED "
        "DIAGNOSTIC REPRODUCTION"
    )
    print("=" * 74)

    print(
        "device:",
        device,
    )

    # ========================================================
    # Metadata
    # ========================================================

    (
        train_metadata,
        val_metadata,
        test_metadata,
    ) = load_class_subset(
        NUM_CLASSES,
        splits_dir=args.splits_dir,
    )

    assert (
        len(val_metadata)
        == EXPECTED_VAL_SIZE
    ), (
        f"Expected {EXPECTED_VAL_SIZE} "
        "validation examples, got "
        f"{len(val_metadata)}."
    )

    print(
        "validation samples:",
        len(val_metadata),
    )

    # ========================================================
    # Correct VideoMAE processor
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

    assert (
        processor_name
        == expected_processor
    )

    print(
        "processor:",
        processor_name,
    )

    # ========================================================
    # Dataset / loader
    # ========================================================

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = build_ssv2_datasets(
        train_metadata,
        val_metadata,
        test_metadata,
        clips_dir=args.clips_dir,
        num_frames=NUM_FRAMES,
        processor_name=processor_name,
    )

    (
        _train_loader,
        val_loader,
        _test_loader,
    ) = build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=TRAINING_SEED,
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
        eval_seed=0,
    ).to(
        device
    )

    # ========================================================
    # Checkpoint
    # ========================================================

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    load_checkpoint_model_state(
        model,
        checkpoint[
            "model_state_dict"
        ],
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

    # ========================================================
    # First verify that the new temporal machinery works on
    # actual VideoMAE output.
    # ========================================================

    check_real_temporal_integration(
        model=model,
        val_loader=val_loader,
        device=device,
    )

    # ========================================================
    # Full validation diagnostic collection
    # ========================================================

    print()
    print("=" * 74)
    print(
        "COLLECTING FULL VALIDATION "
        "DIAGNOSTICS"
    )
    print("=" * 74)

    rows = collect_slot_diagnostics(
        model=model,
        loader=val_loader,
        device=device,
        metadata=val_metadata,
        slot_eval_mode=(
            "per_video_deterministic"
        ),
        slot_eval_seed=SLOT_EVAL_SEED,
        topk_values=[1],
    )

    assert (
        len(rows)
        == EXPECTED_VAL_SIZE
    )

    summary = (
        summarize_diagnostic_rows(
            rows
        )
    )

    # ========================================================
    # Classification reproduction
    # ========================================================

    correct = sum(
        bool(
            row["top1_correct"]
        )
        for row in rows
    )

    top1 = (
        correct
        / len(rows)
    )

    print()
    print("=" * 74)
    print(
        "CLASSIFICATION CHECK"
    )
    print("=" * 74)

    print(
        "correct:",
        f"{correct}/{len(rows)}",
    )

    print(
        "Top-1:",
        f"{top1:.10f}",
    )

    print(
        "canonical:",
        f"{EXPECTED_CORRECT}/"
        f"{EXPECTED_VAL_SIZE}",
        f"= {EXPECTED_TOP1:.10f}",
    )

    assert (
        correct
        == EXPECTED_CORRECT
    ), (
        "Classification reproduction failed: "
        f"got {correct}, "
        f"expected {EXPECTED_CORRECT}."
    )

    assert abs(
        top1
        - EXPECTED_TOP1
    ) < 1e-12

    # ========================================================
    # Diagnostic reproduction
    # ========================================================

    print()
    print("=" * 74)
    print(
        "DIAGNOSTIC MEANS"
    )
    print("=" * 74)

    print(
        f"{'metric':32s} "
        f"{'actual':>12s} "
        f"{'reference':>12s} "
        f"{'difference':>12s}"
    )

    print(
        "-" * 74
    )

    failures = []

    for (
        key,
        expected,
    ) in (
        EXPECTED_DIAGNOSTICS.items()
    ):

        actual = float(
            summary[key]
        )

        difference = abs(
            actual
            - expected
        )

        print(
            f"{key:32s} "
            f"{actual:12.6f} "
            f"{expected:12.6f} "
            f"{difference:12.6f}"
        )

        if (
            not math.isfinite(
                actual
            )
            or difference
            > DIAGNOSTIC_TOLERANCE
        ):
            failures.append(
                (
                    key,
                    actual,
                    expected,
                    difference,
                )
            )

    # ========================================================
    # Additional sanity checks
    # ========================================================

    dominant_counts = [
        0
        for _ in range(
            NUM_SLOTS
        )
    ]

    for row in rows:

        dominant_counts[
            int(
                row[
                    "dominant_soft_slot"
                ]
            )
        ] += 1

    print()
    print(
        "dominant soft-slot counts:",
        dominant_counts,
    )

    print(
        "recorded canonical counts:",
        [79, 64, 76, 78],
    )

    print(
        "sum:",
        sum(dominant_counts),
    )

    assert (
        sum(dominant_counts)
        == EXPECTED_VAL_SIZE
    )

    # We print rather than hard-assert the slot-index counts.
    #
    # Slot identities are exchangeable, so the scientifically
    # meaningful reproduction target is the distributional
    # diagnostic behavior, not a particular slot label.

    # ========================================================
    # Final acceptance
    # ========================================================

    if failures:

        print()
        print(
            "DIAGNOSTIC REPRODUCTION "
            "FAILED:"
        )

        for (
            key,
            actual,
            expected,
            difference,
        ) in failures:

            print(
                f"  {key}: "
                f"actual={actual:.6f}, "
                f"expected≈{expected:.6f}, "
                f"diff={difference:.6f}"
            )

        raise AssertionError(
            "At least one diagnostic mean "
            "fell outside the accepted "
            f"±{DIAGNOSTIC_TOLERANCE} "
            "tolerance."
        )

    print()
    print("=" * 74)
    print(
        "✓ VIDEOMAE K=4 DIAGNOSTIC "
        "REPRODUCTION PASSED"
    )
    print(
        "✓ REAL TEMPORAL DIAGNOSTIC "
        "INTEGRATION PASSED"
    )
    print(
        "✓ PART 7 DIAGNOSTICS / "
        "VISUALIZATION MIGRATION ACCEPTED"
    )
    print("=" * 74)


if __name__ == "__main__":
    main()