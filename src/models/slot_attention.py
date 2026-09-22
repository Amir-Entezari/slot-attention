import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """
    Vanilla Slot Attention.

    Inputs:
        x:          [B, N, D_in]
        token_mask: [B, N] bool, True = valid

    Outputs:
        slots: [B, K, D_slot]
        attn:  [B, K, N]
    """

    def __init__(
        self,
        num_slots: int,
        in_dim: int,
        slot_dim: int = 128,
        iters: int = 3,
        mlp_hidden: int = 256,
        eps: float = 1e-8,
        eval_seed: int = 0,
    ):
        super().__init__()

        self.num_slots = int(num_slots)
        self.in_dim = int(in_dim)
        self.slot_dim = int(slot_dim)
        self.iters = int(iters)
        self.eps = float(eps)

        self.norm_inputs = nn.LayerNorm(
            in_dim
        )
        self.norm_slots = nn.LayerNorm(
            slot_dim
        )
        self.norm_mlp = nn.LayerNorm(
            slot_dim
        )

        self.project_k = nn.Linear(
            in_dim,
            slot_dim,
            bias=False,
        )
        self.project_v = nn.Linear(
            in_dim,
            slot_dim,
            bias=False,
        )
        self.project_q = nn.Linear(
            slot_dim,
            slot_dim,
            bias=False,
        )

        self.gru = nn.GRUCell(
            slot_dim,
            slot_dim,
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                slot_dim,
                mlp_hidden,
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Linear(
                mlp_hidden,
                slot_dim,
            ),
        )

        # Shared learned distribution for
        # vanilla free slots.
        self.slots_mu = nn.Parameter(
            torch.zeros(
                1,
                1,
                slot_dim,
            )
        )

        self.slots_logsigma = nn.Parameter(
            torch.zeros(
                1,
                1,
                slot_dim,
            )
        )

        self.scale = (
            slot_dim ** -0.5
        )

        # Fixed, distinct noise for reproducible
        # legacy evaluation.
        #
        # persistent=False means it does not affect
        # checkpoint compatibility.
        g = torch.Generator()
        g.manual_seed(
            eval_seed
        )

        eval_noise = torch.randn(
            1,
            self.num_slots,
            self.slot_dim,
            generator=g,
        )

        self.register_buffer(
            "_eval_noise",
            eval_noise,
            persistent=False,
        )

    def _initialize_slots(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        mu = self.slots_mu.expand(
            batch_size,
            self.num_slots,
            -1,
        )

        sigma = (
            self.slots_logsigma
            .exp()
            .expand(
                batch_size,
                self.num_slots,
                -1,
            )
        )

        if self.training:
            noise = torch.randn_like(
                mu
            )

        else:
            noise = (
                self._eval_noise
                .expand(
                    batch_size,
                    -1,
                    -1,
                )
            )

        return (
            mu
            + sigma * noise
        )

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        attn_logit_bias: torch.Tensor | None = None,
        initial_slots: torch.Tensor | None = None,
        return_assignment: bool = False,
    ):
        B, N, D = x.shape

        if attn_logit_bias is not None:
            expected_shape = (
                B,
                self.num_slots,
                N,
            )

            if (
                attn_logit_bias.shape
                != expected_shape
            ):
                raise ValueError(
                    "attn_logit_bias shape mismatch: "
                    f"{attn_logit_bias.shape}, "
                    f"expected {expected_shape}"
                )

        if D != self.in_dim:
            raise ValueError(
                f"in_dim mismatch: got {D}, "
                f"expected {self.in_dim}"
            )

        x = self.norm_inputs(
            x
        )

        if token_mask is not None:
            token_mask = token_mask.to(
                dtype=torch.bool
            )

            x = (
                x
                * token_mask
                .unsqueeze(-1)
                .to(x.dtype)
            )

        k = self.project_k(
            x
        )

        v = self.project_v(
            x
        )

        if initial_slots is None:
            slots = (
                self._initialize_slots(
                    B
                )
            )

        else:
            expected_shape = (
                B,
                self.num_slots,
                self.slot_dim,
            )

            if (
                initial_slots.shape
                != expected_shape
            ):
                raise ValueError(
                    "initial_slots shape mismatch: "
                    f"{initial_slots.shape}, "
                    f"expected {expected_shape}"
                )

            slots = initial_slots

        final_assignment = None

        for _ in range(
            self.iters
        ):
            slots_prev = slots

            q = self.project_q(
                self.norm_slots(
                    slots
                )
            )

            attn_logits = (
                torch.einsum(
                    "bkd,bnd->bkn",
                    q,
                    k,
                )
                * self.scale
            )

            if (
                attn_logit_bias
                is not None
            ):
                attn_logits = (
                    attn_logits
                    + attn_logit_bias
                )

            # ------------------------------------
            # Raw token -> slot competition.
            #
            # For every valid token:
            # sum_k assignment[b,k,n] = 1
            # ------------------------------------

            assignment = F.softmax(
                attn_logits,
                dim=1,
            )

            if token_mask is not None:
                assignment_for_return = (
                    assignment
                    * token_mask
                    .unsqueeze(1)
                    .to(
                        assignment.dtype
                    )
                )

            else:
                assignment_for_return = (
                    assignment
                )

            # ------------------------------------
            # Normalize over tokens for Slot
            # updates.
            # ------------------------------------

            attn = (
                assignment
                + self.eps
            )

            if token_mask is not None:
                attn = (
                    attn
                    * token_mask
                    .unsqueeze(1)
                    .to(
                        attn.dtype
                    )
                )

            attn = (
                attn
                / attn.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(
                    self.eps
                )
            )

            updates = torch.einsum(
                "bkn,bnd->bkd",
                attn,
                v,
            )

            slots = (
                self.gru(
                    updates.reshape(
                        B
                        * self.num_slots,
                        self.slot_dim,
                    ),
                    slots_prev.reshape(
                        B
                        * self.num_slots,
                        self.slot_dim,
                    ),
                )
                .reshape(
                    B,
                    self.num_slots,
                    self.slot_dim,
                )
            )

            slots = (
                slots
                + self.mlp(
                    self.norm_mlp(
                        slots
                    )
                )
            )

            final_assignment = (
                assignment_for_return
            )

        if return_assignment:
            return (
                slots,
                attn,
                final_assignment,
            )

        return (
            slots,
            attn,
        )