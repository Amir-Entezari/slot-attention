from src.checkpoints import (
    get_checkpoint_model_state,
    load_checkpoint_model_state,
)

from src.data import (
    SSV2BaselineDataset,
    build_dataloaders,
    build_ssv2_datasets,
    load_class_subset,
    uniform_frame_indices,
)

from src.evaluation import (
    choose_topk,
    evaluate,
    evaluate_slot_per_video_init,
    normalize_topk,
    topk_correct_counts,
)

from src.experiments import (
    resolve_selection_metric,
    run_baseline_experiment,
    run_slot_experiment,
)

from src.models import (
    R3DLayer3BaselineClassifier,
    R3DLayer3SlotAttentionClassifier,
    SlotAttention,
    TimeSformerCLSBaselineClassifier,
    TimeSformerSlotAttentionClassifier,
    TimeSformerVideoBaseline,
    VideoMAEBaselineClassifier,
    VideoMAEControlledBaselineClassifier,
    VideoMAESlotAttentionClassifier,
    VideoResNetBaseline,
    VideoResNetSlotAttentionClassifier,
    ViTSlotAttentionClassifier,
    ViTVideoBaseline,
    build_baseline_model,
    build_slot_model,
    get_baseline_processor_name,
    get_slot_processor_name,
)

from src.training import (
    train_one_epoch,
)

from src.utils import (
    seed_everything,
)


def main():

    # ========================================================
    # Metric policy
    # ========================================================

    assert choose_topk(5) == [1]
    assert choose_topk(20) == [1, 3]
    assert choose_topk(50) == [1, 5]

    assert normalize_topk(
        [1, 3, 5],
        50,
    ) == [1, 3, 5]

    assert resolve_selection_metric(
        None,
        [1, 3],
    ) == "top3"

    assert resolve_selection_metric(
        "top1",
        [1, 3, 5],
    ) == "top1"

    # ========================================================
    # Processor policies
    # ========================================================

    assert (
        get_baseline_processor_name(
            "vit"
        )
        ==
        "google/vit-base-patch16-224"
    )

    # Historical TimeSformer baseline preprocessing.
    assert (
        get_baseline_processor_name(
            "timesformer"
        )
        ==
        "google/vit-base-patch16-224"
    )

    # Corrected TimeSformer preprocessing.
    assert (
        get_baseline_processor_name(
            "timesformer_corrected_mean"
        )
        ==
        "facebook/"
        "timesformer-base-finetuned-k400"
    )

    # Legacy TimeSformer Slot preprocessing.
    assert (
        get_slot_processor_name(
            "timesformer",
            slot_eval_mode=(
                "legacy_shared"
            ),
        )
        ==
        "google/vit-base-patch16-224"
    )

    # Corrected TimeSformer Slot preprocessing.
    assert (
        get_slot_processor_name(
            "timesformer",
            slot_eval_mode=(
                "per_video_deterministic"
            ),
        )
        ==
        "facebook/"
        "timesformer-base-finetuned-k400"
    )

    assert (
        get_slot_processor_name(
            "videomae_controlled",
            slot_eval_mode=(
                "per_video_deterministic"
            ),
        )
        ==
        "MCG-NJU/"
        "videomae-base-finetuned-kinetics"
    )

    # ========================================================
    # Frame sampling
    # ========================================================

    assert uniform_frame_indices(
        100,
        8,
    ) == [
        0,
        14,
        28,
        42,
        56,
        70,
        84,
        99,
    ]

    # ========================================================
    # Lightweight model factories
    #
    # Use R3D so this test does not require Hugging Face
    # downloads.
    # ========================================================

    baseline = build_baseline_model(
        model_name="r3d_18",
        num_classes=5,
        num_frames=8,
    )

    assert isinstance(
        baseline,
        VideoResNetBaseline,
    )

    slot = build_slot_model(
        model_name="r3d_18",
        num_classes=5,
        num_frames=8,
        num_slots=4,
        slot_dim=128,
        slot_iters=3,
    )

    assert isinstance(
        slot,
        VideoResNetSlotAttentionClassifier,
    )

    assert (
        slot.slot_attn.num_slots
        == 4
    )

    assert (
        slot.slot_attn.slot_dim
        == 128
    )

    assert (
        slot.slot_attn.iters
        == 3
    )

    # ========================================================
    # Arbitrary K support
    # ========================================================

    for k in [
        1,
        2,
        4,
        6,
        8,
    ]:
        model = build_slot_model(
            model_name="r3d_18",
            num_classes=5,
            num_frames=8,
            num_slots=k,
        )

        assert (
            model.slot_attn.num_slots
            == k
        )

        assert tuple(
            model
            .slot_attn
            ._eval_noise
            .shape
        ) == (
            1,
            k,
            128,
        )

    print()
    print("=" * 70)
    print(
        "✓ CLEAN REPOSITORY "
        "MIGRATION SMOKE TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()