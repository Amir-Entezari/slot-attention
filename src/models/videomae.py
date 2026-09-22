import torch
import torch.nn as nn

from transformers import (
    VideoMAEForVideoClassification,
)

from .slot_attention import SlotAttention

VIDEOMAE_CHECKPOINT = (
    "MCG-NJU/"
    "videomae-base-finetuned-kinetics"
)


class VideoMAEBaselineClassifier(nn.Module):
    """
    Frozen Kinetics-400-finetuned VideoMAE
        -> native mean-pooled video representation
        -> pretrained fc_norm
        -> 128-D projection
        -> lightweight classifier
        -> SSV2 action logits

    This reproduces the native VideoMAE baseline used in
    the canonical experiment notebook.
    """

    def __init__(
        self,
        num_classes,
        proj_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.checkpoint_name = (
            VIDEOMAE_CHECKPOINT
        )

        self.backbone = (
            VideoMAEForVideoClassification
            .from_pretrained(
                self.checkpoint_name
            )
        )

        if not self.backbone.config.use_mean_pooling:
            raise ValueError(
                "This baseline expects the "
                "VideoMAE checkpoint to use "
                "mean pooling."
            )

        if self.backbone.fc_norm is None:
            raise ValueError(
                "Expected pretrained "
                "VideoMAE fc_norm."
            )

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

        hidden_dim = (
            self.backbone.config.hidden_size
        )

        self.input_proj = nn.Linear(
            hidden_dim,
            proj_dim,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(
                proj_dim,
                256,
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def train(
        self,
        mode=True,
    ):
        super().train(mode)

        # The frozen pretrained backbone must
        # remain deterministic.
        self.backbone.eval()

        return self

    def extract_representation(
        self,
        videos,
    ):
        if videos.ndim != 5:
            raise ValueError(
                "Expected videos "
                "[B,T,C,H,W], "
                f"got {tuple(videos.shape)}"
            )

        expected_frames = int(
            self.backbone.config.num_frames
        )

        if videos.size(1) != expected_frames:
            raise ValueError(
                "VideoMAE frame-count "
                "mismatch: "
                f"got {videos.size(1)}, "
                f"expected "
                f"{expected_frames}"
            )

        with torch.no_grad():

            outputs = (
                self.backbone.videomae(
                    pixel_values=videos
                )
            )

            tokens = (
                outputs.last_hidden_state
            )

            # Reproduce the representation used
            # by VideoMAEForVideoClassification
            # when use_mean_pooling=True.
            video_repr = (
                tokens.mean(
                    dim=1
                )
            )

            video_repr = (
                self.backbone.fc_norm(
                    video_repr
                )
            )

        z = self.input_proj(
            video_repr
        )

        return z

    def forward(
        self,
        videos,
    ):
        z = (
            self.extract_representation(
                videos
            )
        )

        return self.classifier(
            z
        )


class VideoMAEControlledBaselineClassifier(
    nn.Module
):
    """
    Controlled VideoMAE token-mean baseline.

    Pipeline:
        VideoMAE tokens
        -> 768 -> proj_dim projection
        -> mean over projected tokens
        -> same lightweight classifier family

    This is the fair baseline for VideoMAE + Slot:
        SAME backbone
        SAME tokens
        SAME projection
        SAME classifier

    The aggregation operation is what changes:
        mean(tokens)
    versus:
        SlotAttention(tokens) -> mean(slots)
    """

    def __init__(
        self,
        num_classes,
        proj_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.backbone = (
            VideoMAEForVideoClassification
            .from_pretrained(
                VIDEOMAE_CHECKPOINT
            )
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

        hidden_dim = (
            self.backbone.config.hidden_size
        )

        self.input_proj = nn.Linear(
            hidden_dim,
            proj_dim,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                proj_dim
            ),
            nn.Linear(
                proj_dim,
                256,
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def train(
        self,
        mode=True,
    ):
        super().train(mode)

        self.backbone.eval()

        return self

    def extract_projected_tokens(
        self,
        videos,
    ):
        self.backbone.eval()

        with __import__(
            "torch"
        ).no_grad():

            outputs = (
                self.backbone.videomae(
                    pixel_values=videos
                )
            )

            tokens = (
                outputs.last_hidden_state
            )

        return self.input_proj(
            tokens
        )

    def forward(
        self,
        videos,
    ):
        tokens = (
            self.extract_projected_tokens(
                videos
            )
        )

        z = tokens.mean(
            dim=1
        )

        return self.classifier(
            z
        )
        
        

class VideoMAESlotAttentionClassifier(
    VideoMAEControlledBaselineClassifier
):
    def __init__(
        self,
        num_classes,
        num_slots=4,
        slot_dim=128,
        slot_iters=3,
        eval_seed=0,
    ):
        super().__init__(
            num_classes=num_classes,
            proj_dim=slot_dim,
        )

        self.slot_attn = SlotAttention(
            num_slots=num_slots,
            in_dim=slot_dim,
            slot_dim=slot_dim,
            iters=slot_iters,
            eval_seed=eval_seed,
        )

    def forward(
        self,
        videos,
        initial_slots=None,
    ):
        tokens = (
            self.extract_projected_tokens(
                videos
            )
        )

        slots, _attn = self.slot_attn(
            tokens,
            initial_slots=initial_slots,
        )

        z = slots.mean(
            dim=1
        )

        return self.classifier(
            z
        )