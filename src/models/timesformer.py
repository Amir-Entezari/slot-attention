import torch
import torch.nn as nn

from transformers import TimesformerModel

from .slot_attention import SlotAttention

TIMESFORMER_CHECKPOINT = (
    "facebook/timesformer-base-finetuned-k400"
)


class TimeSformerVideoBaseline(nn.Module):
    def __init__(
        self,
        num_classes,
        num_frames,
        repr_dim=128,
    ):
        super().__init__()

        self.backbone = (
            TimesformerModel.from_pretrained(
                TIMESFORMER_CHECKPOINT
            )
        )

        self.num_frames = num_frames

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.input_proj = nn.Linear(
            self.backbone.config.hidden_size,
            repr_dim,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(repr_dim),
            nn.Linear(repr_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        self.backbone.eval()

        with torch.no_grad():
            outputs = self.backbone(
                pixel_values=x
            )

        # Use visual tokens, not CLS.
        tokens = (
            outputs.last_hidden_state[
                :,
                1:,
                :,
            ]
        )

        tokens = self.input_proj(
            tokens
        )

        z = tokens.mean(
            dim=1
        )

        return self.classifier(z)


class TimeSformerCLSBaselineClassifier(nn.Module):
    """
    Frozen pretrained TimeSformer
        -> final CLS token
        -> 128-D projection
        -> same lightweight classifier family
        -> action logits

    This is a backbone-quality reference, not a replacement
    for the controlled token-mean baseline.
    """

    def __init__(
        self,
        num_classes,
        proj_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.backbone = (
            TimesformerModel.from_pretrained(
                TIMESFORMER_CHECKPOINT
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
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def train(self, mode=True):
        """
        Keep the frozen TimeSformer backbone in eval mode
        even while the classifier is training.
        """
        super().train(mode)

        self.backbone.eval()

        return self

    def extract_representation(
        self,
        videos,
    ):
        with torch.no_grad():
            outputs = self.backbone(
                pixel_values=videos
            )

            # Pretrained video-level representation.
            cls_token = (
                outputs.last_hidden_state[
                    :,
                    0,
                    :,
                ]
            )

        z = self.input_proj(
            cls_token
        )

        return z

    def forward(
        self,
        videos,
    ):
        z = self.extract_representation(
            videos
        )

        return self.classifier(z)
    
    

class TimeSformerSlotAttentionClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        num_slots=4,
        slot_dim=128,
        slot_iters=3,
        eval_seed=0,
    ):
        super().__init__()

        self.backbone = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        in_dim = self.backbone.config.hidden_size

        self.input_proj = nn.Linear(
            in_dim,
            slot_dim,
        )

        self.slot_attn = SlotAttention(
            num_slots=num_slots,
            in_dim=slot_dim,
            slot_dim=slot_dim,
            iters=slot_iters,
            eval_seed=eval_seed,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def extract_projected_tokens(
        self,
        x,
    ):
        self.backbone.eval()

        with torch.no_grad():
            outputs = self.backbone(
                pixel_values=x
            )

        tokens = (
            outputs
            .last_hidden_state[
                :,
                1:,
                :,
            ]
        )

        return self.input_proj(
            tokens
        )

    def forward(
        self,
        x,
        initial_slots=None,
    ):
        tokens = (
            self.extract_projected_tokens(
                x
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