import torch.nn as nn

from torchvision.models.video import (
    mc3_18,
    r3d_18,
)

from .slot_attention import SlotAttention


class VideoResNetBaseline(nn.Module):
    """
    Canonical R3D-18 / MC3-18 controlled baseline.

    Pipeline:
        video [B,T,C,H,W]
        -> torchvision video backbone
        -> layer4 feature grid
        -> flatten [T,H,W] into tokens
        -> project 512 -> repr_dim
        -> mean token pooling
        -> lightweight classifier

    The backbone is trained from scratch, matching the
    canonical experiment notebook.
    """

    def __init__(
        self,
        num_classes,
        backbone_name="r3d_18",
        repr_dim=128,
    ):
        super().__init__()

        if backbone_name == "r3d_18":
            self.backbone = r3d_18(
                weights=None
            )

        elif backbone_name == "mc3_18":
            self.backbone = mc3_18(
                weights=None
            )

        else:
            raise ValueError(
                f"Unsupported backbone_name: "
                f"{backbone_name}"
            )

        self.backbone.fc = nn.Identity()

        self.input_proj = nn.Linear(
            512,
            repr_dim,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(repr_dim),
            nn.Linear(repr_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def extract_tokens(
        self,
        x,
    ):
        # Dataset:
        # [B,T,C,H,W]
        #
        # torchvision video models:
        # [B,C,T,H,W]
        x = x.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        b, c, t, h, w = x.shape

        tokens = (
            x.permute(
                0, 2, 3, 4, 1
            )
            .contiguous()
            .view(
                b,
                t * h * w,
                c,
            )
        )

        return tokens

    def forward(
        self,
        x,
    ):
        tokens = self.extract_tokens(
            x
        )

        tokens = self.input_proj(
            tokens
        )

        z = tokens.mean(
            dim=1
        )

        return self.classifier(
            z
        )


class R3DLayer3BaselineClassifier(nn.Module):
    """
    R3D-18 temporal-feature-stage audit.

    Instead of extracting layer4 features, this model stops
    at layer3, where the 8-frame input still has an explicit
    temporal extent of approximately T=2.

    This is an audit model, not the canonical R3D baseline.
    """

    def __init__(
        self,
        num_classes,
        repr_dim=128,
    ):
        super().__init__()

        self.backbone = r3d_18(
            weights=None
        )

        # layer3 output channels = 256
        self.input_proj = nn.Linear(
            256,
            repr_dim,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(repr_dim),
            nn.Linear(repr_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def extract_tokens(
        self,
        x,
    ):
        # Dataset:
        # [B,T,C,H,W]
        #
        # R3D:
        # [B,C,T,H,W]
        x = x.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)

        b, c, t, h, w = x.shape

        tokens = (
            x.permute(
                0, 2, 3, 4, 1
            )
            .contiguous()
            .view(
                b,
                t * h * w,
                c,
            )
        )

        return tokens

    def extract_projected_tokens(
        self,
        x,
    ):
        return self.input_proj(
            self.extract_tokens(x)
        )

    def forward(
        self,
        x,
    ):
        tokens = (
            self.extract_projected_tokens(
                x
            )
        )

        z = tokens.mean(
            dim=1
        )

        return self.classifier(
            z
        )
        
        
        
class VideoResNetSlotAttentionClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_name="r3d_18",
        num_slots=4,
        slot_dim=128,
        slot_iters=3,
        eval_seed=0,
    ):
        super().__init__()

        if backbone_name == "r3d_18":
            self.backbone = r3d_18(
                weights=None
            )

        elif backbone_name == "mc3_18":
            self.backbone = mc3_18(
                weights=None
            )

        else:
            raise ValueError(
                f"Unsupported backbone_name: "
                f"{backbone_name}"
            )

        self.backbone.fc = nn.Identity()

        self.input_proj = nn.Linear(
            512,
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

    def extract_tokens(
        self,
        x,
    ):
        # Input:
        # [B,T,C,H,W]
        #
        # torchvision video models expect:
        # [B,C,T,H,W]
        x = x.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        b, c, t, h, w = x.shape

        tokens = (
            x.permute(
                0, 2, 3, 4, 1
            )
            .contiguous()
            .view(
                b,
                t * h * w,
                c,
            )
        )

        return tokens

    def extract_projected_tokens(
        self,
        x,
    ):
        return self.input_proj(
            self.extract_tokens(x)
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


class R3DLayer3SlotAttentionClassifier(
    R3DLayer3BaselineClassifier
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
            repr_dim=slot_dim,
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
        x,
    ):
        tokens = (
            self.extract_projected_tokens(
                x
            )
        )

        slots, _attn = self.slot_attn(
            tokens
        )

        z = slots.mean(
            dim=1
        )

        return self.classifier(
            z
        )