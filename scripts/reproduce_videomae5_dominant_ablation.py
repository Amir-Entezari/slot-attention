import argparse

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
    evaluate_dominant_slot_ablation,
)

from src.models import (
    build_slot_model,
    get_slot_processor_name,
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
        default="cuda",
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
    )

    # ========================================================
    # Canonical corrected VideoMAE K=4 configuration
    # ========================================================

    num_classes = 5
    num_frames = 16

    (
        train_metadata,
        val_metadata,
        test_metadata,
    ) = load_class_subset(
        num_classes,
        splits_dir=args.splits_dir,
    )

    processor_name = (
        get_slot_processor_name(
            "videomae_controlled",
            slot_eval_mode=(
                "per_video_deterministic"
            ),
        )
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = build_ssv2_datasets(
        train_metadata,
        val_metadata,
        test_metadata,
        clips_dir=args.clips_dir,
        num_frames=num_frames,
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
        batch_size=2,
        num_workers=2,
        seed=42,
    )

    model = build_slot_model(
        model_name=(
            "videomae_controlled"
        ),
        num_classes=num_classes,
        num_frames=num_frames,
        num_slots=4,
        slot_dim=128,
        slot_iters=3,
        eval_seed=0,
    ).to(
        device
    )

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

    result = (
        evaluate_dominant_slot_ablation(
            model=model,
            loader=val_loader,
            device=device,
            slot_eval_mode=(
                "per_video_deterministic"
            ),
            slot_eval_seed=12345,
        )
    )

    print()
    print("=" * 70)
    print(
        "VIDEOMAE K=4 "
        "DOMINANT-SLOT ABLATION"
    )
    print("=" * 70)

    print(
        "N:",
        result["num_examples"],
    )

    print(
        "full:",
        f"{result['full_correct']}"
        f"/{result['num_examples']}",
        f"= {result['full_top1']:.4f}",
    )

    print(
        "dominant only:",
        f"{result['dominant_correct']}"
        f"/{result['num_examples']}",
        f"= {result['dominant_top1']:.4f}",
    )

    print(
        "Top-1 drop:",
        f"{100 * result['top1_drop']:.2f} pp",
    )

    print(
        "prediction flips:",
        result["prediction_flips"],
        f"({100 * result['prediction_flip_rate']:.2f}%)",
    )

    print(
        "full correct / dominant wrong:",
        result[
            "full_correct_dominant_wrong"
        ],
    )

    print(
        "full wrong / dominant correct:",
        result[
            "full_wrong_dominant_correct"
        ],
    )

    print(
        "dominant slot counts:",
        result[
            "dominant_slot_counts"
        ],
    )

    # ========================================================
    # Canonical reproduction assertions
    # ========================================================

    assert (
        result["num_examples"]
        == 297
    )

    assert (
        result["full_correct"]
        == 214
    )

    assert (
        result["dominant_correct"]
        == 161
    )

    assert (
        result[
            "full_correct_dominant_wrong"
        ]
        == 64
    )

    assert (
        result[
            "full_wrong_dominant_correct"
        ]
        == 11
    )

    print()
    print(
        "✓ DOMINANT-SLOT FUNCTIONAL "
        "ABLATION REPRODUCED"
    )


if __name__ == "__main__":
    main()