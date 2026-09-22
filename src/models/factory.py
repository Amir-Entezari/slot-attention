from .timesformer import (
    TimeSformerCLSBaselineClassifier,
    TimeSformerSlotAttentionClassifier,
    TimeSformerVideoBaseline,
)
from .video_resnet import (
    R3DLayer3BaselineClassifier,
    R3DLayer3SlotAttentionClassifier,
    VideoResNetBaseline,
    VideoResNetSlotAttentionClassifier,
)
from .videomae import (
    VideoMAEBaselineClassifier,
    VideoMAEControlledBaselineClassifier,
    VideoMAESlotAttentionClassifier,
)
from .vit import (
    ViTSlotAttentionClassifier,
    ViTVideoBaseline,
)


VIT_PROCESSOR = (
    "google/vit-base-patch16-224"
)

TIMESFORMER_PROCESSOR = (
    "facebook/timesformer-base-finetuned-k400"
)

VIDEOMAE_PROCESSOR = (
    "MCG-NJU/"
    "videomae-base-finetuned-kinetics"
)


# ============================================================
# Baseline models
# ============================================================

def get_baseline_processor_name(
    model_name,
):
    """
    Return the processor used by the canonical notebook
    for a baseline experiment.

    Historical detail:
        `timesformer` used the ViT processor.

    Corrected variants:
        timesformer_corrected_mean
        timesformer_corrected_cls

    use the proper TimeSformer processor.
    """

    if model_name.startswith(
        "videomae"
    ):
        return VIDEOMAE_PROCESSOR

    if model_name.startswith(
        "timesformer_corrected"
    ):
        return TIMESFORMER_PROCESSOR

    return VIT_PROCESSOR


def build_baseline_model(
    model_name,
    num_classes,
    num_frames,
):
    if model_name == "vit":
        return ViTVideoBaseline(
            num_classes=num_classes,
        )

    if model_name == "timesformer":
        return TimeSformerVideoBaseline(
            num_classes=num_classes,
            num_frames=num_frames,
        )

    if (
        model_name
        == "timesformer_corrected_mean"
    ):
        return TimeSformerVideoBaseline(
            num_classes=num_classes,
            num_frames=num_frames,
        )

    if (
        model_name
        == "timesformer_corrected_cls"
    ):
        return (
            TimeSformerCLSBaselineClassifier(
                num_classes=num_classes,
            )
        )

    if model_name == "r3d_18":
        return VideoResNetBaseline(
            num_classes=num_classes,
            backbone_name="r3d_18",
        )

    if model_name == "mc3_18":
        return VideoResNetBaseline(
            num_classes=num_classes,
            backbone_name="mc3_18",
        )

    if model_name == "r3d_18_layer3":
        return (
            R3DLayer3BaselineClassifier(
                num_classes=num_classes,
                repr_dim=128,
            )
        )

    if model_name == "videomae":
        return (
            VideoMAEBaselineClassifier(
                num_classes=num_classes,
                proj_dim=128,
            )
        )

    if model_name == "videomae_controlled":
        return (
            VideoMAEControlledBaselineClassifier(
                num_classes=num_classes,
                proj_dim=128,
            )
        )

    raise ValueError(
        "Unsupported baseline model_name: "
        f"{model_name}"
    )


# ============================================================
# Slot Attention models
# ============================================================

def get_slot_processor_name(
    model_name,
    slot_eval_mode="legacy_shared",
):
    """
    Reproduce the processor-selection behavior from the
    canonical Slot experiment runner.

    Historical protocol:
        ViT / R3D / MC3 / legacy TimeSformer Slot
        use the ViT processor.

    Corrected TimeSformer protocol:
        TimeSformer + per-video deterministic evaluation
        uses the proper TimeSformer processor.

    VideoMAE always uses its own processor.
    """

    if model_name.startswith(
        "videomae"
    ):
        return VIDEOMAE_PROCESSOR

    if (
        model_name == "timesformer"
        and slot_eval_mode
        == "per_video_deterministic"
    ):
        return TIMESFORMER_PROCESSOR

    return VIT_PROCESSOR


def build_slot_model(
    model_name,
    num_classes,
    num_frames,
    num_slots=4,
    slot_dim=128,
    slot_iters=3,
    eval_seed=0,
):
    """
    Build one of the canonical Slot Attention models.

    `num_frames` is kept in the signature for compatibility
    with the canonical experiment API. The currently migrated
    wrappers infer their token geometry from the input.
    """

    if model_name == "vit":
        return ViTSlotAttentionClassifier(
            num_classes=num_classes,
            num_slots=num_slots,
            slot_dim=slot_dim,
            slot_iters=slot_iters,
            eval_seed=eval_seed,
        )

    if model_name == "timesformer":
        return (
            TimeSformerSlotAttentionClassifier(
                num_classes=num_classes,
                num_slots=num_slots,
                slot_dim=slot_dim,
                slot_iters=slot_iters,
                eval_seed=eval_seed,
            )
        )

    if model_name == "r3d_18":
        return (
            VideoResNetSlotAttentionClassifier(
                num_classes=num_classes,
                backbone_name="r3d_18",
                num_slots=num_slots,
                slot_dim=slot_dim,
                slot_iters=slot_iters,
                eval_seed=eval_seed,
            )
        )

    if model_name == "mc3_18":
        return (
            VideoResNetSlotAttentionClassifier(
                num_classes=num_classes,
                backbone_name="mc3_18",
                num_slots=num_slots,
                slot_dim=slot_dim,
                slot_iters=slot_iters,
                eval_seed=eval_seed,
            )
        )

    if model_name == "r3d_18_layer3":
        return (
            R3DLayer3SlotAttentionClassifier(
                num_classes=num_classes,
                num_slots=num_slots,
                slot_dim=slot_dim,
                slot_iters=slot_iters,
                eval_seed=eval_seed,
            )
        )

    if model_name == "videomae_controlled":
        return (
            VideoMAESlotAttentionClassifier(
                num_classes=num_classes,
                num_slots=num_slots,
                slot_dim=slot_dim,
                slot_iters=slot_iters,
                eval_seed=eval_seed,
            )
        )

    raise ValueError(
        "Unsupported Slot model_name: "
        f"{model_name}"
    )