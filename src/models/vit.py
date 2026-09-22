import torch
import torch.nn as nn

from transformers import ViTModel

from .slot_attention import SlotAttention

class ViTVideoBaseline(nn.Module):
    """
    Canonical frozen-ViT controlled baseline.

    Pipeline:
        video [B,T,C,H,W]
        -> process each frame independently with ViT
        -> remove CLS tokens
        -> concatenate patch tokens across frames
        -> project 768 -> repr_dim
        -> mean-pool all visual tokens
        -> lightweight classifier
    """

    def __init__(
        self,
        num_classes,
        repr_dim=128,
    ):
        super().__init__()

        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224"
        )

        for param in self.vit.parameters():
            param.requires_grad = False

        self.input_proj = nn.Linear(
            self.vit.config.hidden_size,
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
        b, t, c, h, w = x.shape

        x = x.view(
            b * t,
            c,
            h,
            w,
        )

        # Frozen feature extractor.
        self.vit.eval()

        with torch.no_grad():
            outputs = self.vit(
                pixel_values=x
            )

        # Use patch tokens only.
        # These are the exact same visual tokens used by
        # the canonical ViT + Slot model.
        tokens = (
            outputs.last_hidden_state[
                :,
                1:,
                :
            ]
        )

        tokens = tokens.view(
            b,
            t,
            tokens.size(1),
            tokens.size(2),
        )

        tokens = tokens.reshape(
            b,
            t * tokens.size(2),
            tokens.size(3),
        )

        tokens = self.input_proj(
            tokens
        )

        # Controlled baseline aggregation.
        z = tokens.mean(
            dim=1
        )

        return self.classifier(z)
    
    
    
    
    
    
class ViTSlotAttentionClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        num_slots=4,
        slot_dim=128,
        slot_iters=3,
        eval_seed=0,
    ):
        super().__init__()

        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224"
        )

        for param in self.vit.parameters():
            param.requires_grad = False

        in_dim = self.vit.config.hidden_size

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
            nn.Linear(
                slot_dim,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def extract_projected_tokens(
        self,
        x,
    ):
        b, t, c, h, w = x.shape

        x = x.view(
            b * t,
            c,
            h,
            w,
        )

        self.vit.eval()

        with torch.no_grad():
            outputs = self.vit(
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

        tokens = tokens.view(
            b,
            t,
            tokens.size(1),
            tokens.size(2),
        )

        tokens = tokens.reshape(
            b,
            t * tokens.size(2),
            tokens.size(3),
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