import math

import torch
import torch.nn.functional as F

from torch.utils.data import SequentialSampler

def assignment_statistics(
    assignment,
    eps=1e-8,
):
    """
    Compute per-example Slot Attention assignment statistics.

    Parameters
    ----------
    assignment : torch.Tensor
        Raw token-to-slot competition probabilities:

            P(slot | token)

        Shape:
            [B, K, N]

        where:
            B = batch size
            K = number of slots
            N = number of tokens

        Probabilities must sum to approximately 1 over K
        for every token.

    eps : float
        Numerical stability constant.

    Returns
    -------
    dict[str, torch.Tensor]

        Scalar metrics have shape [B].

        Per-slot usage tensors have shape [B, K].
    """

    if assignment.ndim != 3:
        raise ValueError(
            "assignment must have shape "
            "[B, K, N]. "
            f"Got {tuple(assignment.shape)}."
        )

    batch_size, num_slots, num_tokens = (
        assignment.shape
    )

    if num_slots < 1:
        raise ValueError(
            "assignment must contain at least "
            "one slot."
        )

    if num_tokens < 1:
        raise ValueError(
            "assignment must contain at least "
            "one token."
        )

    if not torch.isfinite(
        assignment
    ).all():
        raise ValueError(
            "assignment contains non-finite values."
        )

    if (
        assignment < -1e-6
    ).any():
        raise ValueError(
            "assignment contains negative probabilities."
        )

    slot_sums = assignment.sum(
        dim=1
    )

    if not torch.allclose(
        slot_sums,
        torch.ones_like(
            slot_sums
        ),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError(
            "assignment must sum to 1 over "
            "the slot dimension for every token."
        )

    # Clamp only for log operations.
    p = assignment.clamp_min(
        eps
    )

    # ========================================================
    # 1. Assignment entropy
    #
    # Entropy of P(slot | token), averaged over tokens.
    #
    # Normalize by log(K), giving:
    #
    #     0 -> fully decisive assignment
    #     1 -> uniform assignment over slots
    # ========================================================

    if num_slots == 1:
        assign_entropy_mean = torch.zeros(
            batch_size,
            device=assignment.device,
            dtype=assignment.dtype,
        )

    else:
        per_token_entropy = -(
            assignment
            * p.log()
        ).sum(
            dim=1
        )

        per_token_entropy = (
            per_token_entropy
            / math.log(
                num_slots
            )
        )

        assign_entropy_mean = (
            per_token_entropy.mean(
                dim=1
            )
        )

    # ========================================================
    # 2. Assignment confidence
    #
    # For every token:
    #     max_k P(slot=k | token)
    #
    # Then average over tokens.
    # ========================================================

    assignment_confidence = (
        assignment
        .max(dim=1)
        .values
        .mean(dim=1)
    )

    # ========================================================
    # 3. Soft slot usage
    #
    # Average assignment probability received by each slot.
    #
    # Shape:
    #     [B, K]
    #
    # Sum over slots = 1.
    # ========================================================

    soft_usage = assignment.mean(
        dim=2
    )

    soft_usage_safe = (
        soft_usage.clamp_min(
            eps
        )
    )

    # ========================================================
    # 4. Usage entropy and effective number of slots
    #
    # IMPORTANT:
    # usage_entropy is NOT normalized by log(K).
    #
    # effective_slots = exp(H(soft_usage))
    #
    # Range:
    #     1 <= effective_slots <= K
    # ========================================================

    usage_entropy_raw = -(
        usage
        * usage_safe.log()
    ).sum(dim=1)


    if num_slots == 1:
        usage_entropy = torch.zeros_like(
            usage_entropy_raw
        )

    else:
        usage_entropy = (
            usage_entropy_raw
            / math.log(num_slots)
        )

    effective_slots = (
        usage_entropy.exp()
    )

    soft_usage_min = (
        soft_usage.min(
            dim=1
        ).values
    )

    soft_usage_max = (
        soft_usage.max(
            dim=1
        ).values
    )

    # ========================================================
    # 5. Hard slot usage
    #
    # Winner for each token:
    #
    #     argmax_k P(slot=k | token)
    #
    # Convert winners to fractions of tokens assigned to
    # each slot.
    # ========================================================

    hard_winner = assignment.argmax(
        dim=1
    )

    hard_usage = (
        torch.nn.functional.one_hot(
            hard_winner,
            num_classes=num_slots,
        )
        .to(
            dtype=assignment.dtype
        )
        .mean(dim=1)
    )

    hard_usage_min = (
        hard_usage.min(
            dim=1
        ).values
    )

    hard_usage_max = (
        hard_usage.max(
            dim=1
        ).values
    )

    # Number of slots receiving zero hard-winning tokens.
    hard_dead_slots = (
        hard_usage
        .eq(0)
        .sum(dim=1)
        .to(
            dtype=assignment.dtype
        )
    )

    return {
        "assign_entropy_mean":
            assign_entropy_mean,

        "assignment_confidence":
            assignment_confidence,

        "usage_entropy":
            usage_entropy,

        "effective_slots":
            effective_slots,

        "soft_usage":
            soft_usage,

        "soft_usage_min":
            soft_usage_min,

        "soft_usage_max":
            soft_usage_max,

        "hard_usage":
            hard_usage,

        "hard_usage_min":
            hard_usage_min,

        "hard_usage_max":
            hard_usage_max,

        "hard_dead_slots":
            hard_dead_slots,
    }
    
    
    

def _pairwise_upper_triangle(
    matrix,
):
    """
    Extract unique off-diagonal pairwise values.

    Input
    -----
    matrix:
        [B, K, K]

    Returns
    -------
    [B, K*(K-1)/2]
    """

    if matrix.ndim != 3:
        raise ValueError(
            "matrix must have shape [B, K, K]."
        )

    _, k1, k2 = matrix.shape

    if k1 != k2:
        raise ValueError(
            "pairwise matrix must be square."
        )

    mask = torch.triu(
        torch.ones(
            k1,
            k1,
            dtype=torch.bool,
            device=matrix.device,
        ),
        diagonal=1,
    )

    return matrix[:, mask]


def representation_attention_statistics(
    slots,
    attn,
    eps=1e-8,
):
    """
    Compute slot-vector and token-attention diagnostics.

    Parameters
    ----------
    slots:
        Final Slot Attention vectors.

        Shape:
            [B, K, D]

    attn:
        Per-slot token distributions returned by SlotAttention.

        Shape:
            [B, K, N]

        Each slot must approximately sum to 1 over N.

        IMPORTANT:
            This is NOT P(slot | token).
            It is the later token-normalized attention:

                P(token | slot)

            conceptually.

    Returns
    -------
    dict[str, torch.Tensor]

    Scalar metrics have shape [B].

    Per-slot token entropies have shape [B, K].
    """

    # ========================================================
    # Validation
    # ========================================================

    if slots.ndim != 3:
        raise ValueError(
            "slots must have shape [B, K, D]. "
            f"Got {tuple(slots.shape)}."
        )

    if attn.ndim != 3:
        raise ValueError(
            "attn must have shape [B, K, N]. "
            f"Got {tuple(attn.shape)}."
        )

    if (
        slots.shape[0]
        != attn.shape[0]
    ):
        raise ValueError(
            "slots and attn batch sizes differ."
        )

    if (
        slots.shape[1]
        != attn.shape[1]
    ):
        raise ValueError(
            "slots and attn must contain "
            "the same number of slots."
        )

    batch_size = slots.shape[0]
    num_slots = slots.shape[1]
    num_tokens = attn.shape[2]

    if num_slots < 1:
        raise ValueError(
            "At least one slot is required."
        )

    if num_tokens < 1:
        raise ValueError(
            "At least one token is required."
        )

    if not torch.isfinite(
        slots
    ).all():
        raise ValueError(
            "slots contains non-finite values."
        )

    if not torch.isfinite(
        attn
    ).all():
        raise ValueError(
            "attn contains non-finite values."
        )

    if (
        attn < -1e-6
    ).any():
        raise ValueError(
            "attn contains negative probabilities."
        )

    attn_sums = attn.sum(
        dim=-1
    )

    if not torch.allclose(
        attn_sums,
        torch.ones_like(
            attn_sums
        ),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError(
            "attn must sum to 1 over tokens "
            "for every slot."
        )

    # ========================================================
    # 1. Pairwise slot-vector cosine similarity
    #
    # Compare final slot representations:
    #
    #     slot_i ∈ R^D
    #
    # High cosine similarity means final slot vectors point
    # in similar representation-space directions.
    # ========================================================

    slot_norm = F.normalize(
        slots,
        p=2,
        dim=-1,
        eps=eps,
    )

    slot_cos_matrix = torch.matmul(
        slot_norm,
        slot_norm.transpose(
            1,
            2,
        ),
    )

    slot_cos_pairs = (
        _pairwise_upper_triangle(
            slot_cos_matrix
        )
    )

    # ========================================================
    # 2. Pairwise attention cosine similarity
    #
    # Compare each slot's distribution over tokens.
    # ========================================================

    attn_l2_norm = F.normalize(
        attn,
        p=2,
        dim=-1,
        eps=eps,
    )

    attn_cos_matrix = torch.matmul(
        attn_l2_norm,
        attn_l2_norm.transpose(
            1,
            2,
        ),
    )

    attn_cos_pairs = (
        _pairwise_upper_triangle(
            attn_cos_matrix
        )
    )

    # ========================================================
    # 3. Pairwise Jensen-Shannon divergence
    #
    # For slots i,j:
    #
    #     M = 0.5 * (P_i + P_j)
    #
    #     JS(P_i,P_j)
    #       = 0.5 KL(P_i || M)
    #       + 0.5 KL(P_j || M)
    #
    # Natural logarithms are used.
    #
    # Therefore:
    #
    #     0 <= JS <= log(2)
    #
    # We do NOT normalize by log(2).
    # ========================================================

    p = attn.unsqueeze(
        dim=2
    )

    q = attn.unsqueeze(
        dim=1
    )

    m = 0.5 * (
        p + q
    )

    p_safe = p.clamp_min(
        eps
    )

    q_safe = q.clamp_min(
        eps
    )

    m_safe = m.clamp_min(
        eps
    )

    kl_pm = (
        p
        * (
            p_safe.log()
            - m_safe.log()
        )
    ).sum(
        dim=-1
    )

    kl_qm = (
        q
        * (
            q_safe.log()
            - m_safe.log()
        )
    ).sum(
        dim=-1
    )

    js_matrix = 0.5 * (
        kl_pm + kl_qm
    )

    attn_js_pairs = (
        _pairwise_upper_triangle(
            js_matrix
        )
        / math.log(2.0)
    )

    # ========================================================
    # 4. Per-slot token-distribution entropy
    #
    # Entropy of each slot's attention over N tokens.
    #
    # Normalize by log(N):
    #
    #     0 -> concentrated on one token
    #     1 -> uniform over all tokens
    # ========================================================

    if num_tokens == 1:
        slot_token_entropy = (
            torch.zeros(
                batch_size,
                num_slots,
                dtype=attn.dtype,
                device=attn.device,
            )
        )

    else:
        attn_safe = (
            attn.clamp_min(
                eps
            )
        )

        slot_token_entropy = -(
            attn
            * attn_safe.log()
        ).sum(
            dim=-1
        )

        slot_token_entropy = (
            slot_token_entropy
            / math.log(
                num_tokens
            )
        )

    # ========================================================
    # K=1
    #
    # Pairwise statistics are mathematically undefined because
    # there is no pair of slots.
    #
    # Return NaN rather than pretending similarity/divergence
    # is zero.
    # ========================================================

    if num_slots == 1:

        nan_values = torch.full(
            (batch_size,),
            float("nan"),
            dtype=slots.dtype,
            device=slots.device,
        )

        slot_cos_mean = (
            nan_values.clone()
        )

        slot_cos_max = (
            nan_values.clone()
        )

        attn_cos_mean = (
            nan_values.clone()
        )

        attn_cos_max = (
            nan_values.clone()
        )

        attn_js_mean = (
            nan_values.clone()
        )

    else:

        slot_cos_mean = (
            slot_cos_pairs.mean(
                dim=1
            )
        )

        slot_cos_max = (
            slot_cos_pairs.max(
                dim=1
            ).values
        )

        attn_cos_mean = (
            attn_cos_pairs.mean(
                dim=1
            )
        )

        attn_cos_max = (
            attn_cos_pairs.max(
                dim=1
            ).values
        )

        attn_js_mean = (
            attn_js_pairs.mean(
                dim=1
            )
        )

    return {
        "slot_cos_mean":
            slot_cos_mean,

        "slot_cos_max":
            slot_cos_max,

        "attn_cos_mean":
            attn_cos_mean,

        "attn_cos_max":
            attn_cos_max,

        "attn_js_mean":
            attn_js_mean,

        "slot_token_entropy":
            slot_token_entropy,

        "slot_token_entropy_mean":
            slot_token_entropy.mean(
                dim=1
            ),

        "slot_token_entropy_min":
            slot_token_entropy.min(
                dim=1
            ).values,

        "slot_token_entropy_max":
            slot_token_entropy.max(
                dim=1
            ).values,
    }
    
    
def _metadata_value(
    row,
    key,
    default=None,
):
    """
    Read a value from a metadata row.

    Supports dict-like rows and simple objects.
    """

    if row is None:
        return default

    if isinstance(
        row,
        dict,
    ):
        return row.get(
            key,
            default,
        )

    if hasattr(
        row,
        "get",
    ):
        try:
            return row.get(
                key,
                default,
            )
        except Exception:
            pass

    return getattr(
        row,
        key,
        default,
    )


def _build_per_video_initial_slots(
    model,
    noise,
):
    """
    Convert deterministic standard-normal noise into
    SlotAttention initial slots using the model's learned
    mu / logsigma parameters.
    """

    batch_size = noise.shape[0]

    num_slots = (
        model.slot_attn.num_slots
    )

    mu = (
        model
        .slot_attn
        .slots_mu
        .expand(
            batch_size,
            num_slots,
            -1,
        )
    )

    sigma = (
        model
        .slot_attn
        .slots_logsigma
        .exp()
        .expand(
            batch_size,
            num_slots,
            -1,
        )
    )

    noise = noise.to(
        device=mu.device,
        dtype=mu.dtype,
    )

    return (
        mu
        + sigma * noise
    )


def _official_prediction_forward(
    model,
    videos,
    initial_slots,
    device,
):
    """
    Classification forward matching the official evaluator's
    AMP/autocast behavior.
    """

    with (
        torch.inference_mode(),
        torch.cuda.amp.autocast(
            enabled=(
                device.type == "cuda"
            )
        ),
    ):

        if initial_slots is None:
            return model(
                videos
            )

        try:
            return model(
                videos,
                initial_slots=initial_slots,
            )

        except TypeError as exc:
            raise TypeError(
                "This model wrapper does not accept "
                "external initial_slots, so corrected "
                "per-video diagnostic evaluation is "
                "not available for this wrapper."
            ) from exc


def collect_slot_diagnostics(
    model,
    loader,
    device,
    *,
    metadata=None,
    slot_eval_mode="legacy_shared",
    slot_eval_seed=12345,
    topk_values=(1,),
):
    """
    Collect one diagnostic row per dataset example.

    Parameters
    ----------
    model:
        Slot classifier exposing:

            model.extract_projected_tokens(...)
            model.slot_attn
            model.classifier

    loader:
        Evaluation DataLoader.

        Must use deterministic sequential ordering because
        rows are aligned with metadata and, under corrected
        evaluation, with the deterministic initialization
        bank.

    device:
        CPU or CUDA device.

    metadata:
        Optional sequence aligned 1:1 with loader.dataset.

        If supplied, video_id and original label information
        are copied into each output row when available.

    slot_eval_mode:
        "legacy_shared"
        or
        "per_video_deterministic"

    slot_eval_seed:
        Seed used to generate the corrected [N,K,D]
        initialization bank.

    topk_values:
        Top-k correctness fields to store per example.

    Returns
    -------
    list[dict]
        One row per dataset example.
    """

    device = torch.device(
        device
    )

    valid_modes = {
        "legacy_shared",
        "per_video_deterministic",
    }

    if (
        slot_eval_mode
        not in valid_modes
    ):
        raise ValueError(
            "slot_eval_mode must be one of "
            f"{sorted(valid_modes)}."
        )

    if not hasattr(
        model,
        "slot_attn",
    ):
        raise ValueError(
            "model has no slot_attn module."
        )

    if not hasattr(
        model,
        "extract_projected_tokens",
    ):
        raise ValueError(
            "model must expose "
            "extract_projected_tokens()."
        )

    if not isinstance(
        loader.sampler,
        SequentialSampler,
    ):
        raise ValueError(
            "Diagnostic collection requires "
            "shuffle=False / SequentialSampler."
        )

    num_examples = len(
        loader.dataset
    )

    if (
        metadata is not None
        and len(metadata)
        != num_examples
    ):
        raise ValueError(
            "metadata length must equal "
            "len(loader.dataset)."
        )

    topk_values = sorted(
        {
            int(k)
            for k in topk_values
        }
    )

    if (
        not topk_values
        or topk_values[0] < 1
    ):
        raise ValueError(
            "topk_values must contain "
            "positive integers."
        )

    model.eval()

    # ========================================================
    # Corrected initialization bank
    # ========================================================

    noise_bank = None

    if (
        slot_eval_mode
        == "per_video_deterministic"
    ):
        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            slot_eval_seed
        )

        noise_bank = torch.randn(
            num_examples,
            model.slot_attn.num_slots,
            model.slot_attn.slot_dim,
            generator=generator,
        )

    rows = []

    offset = 0

    # ========================================================
    # Dataset pass
    # ========================================================

    for videos, labels in loader:

        batch_size = labels.size(0)

        videos = videos.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Initial slots
        # ----------------------------------------------------

        if noise_bank is None:
            initial_slots = None

        else:
            noise = noise_bank[
                offset:
                offset + batch_size
            ]

            initial_slots = (
                _build_per_video_initial_slots(
                    model,
                    noise,
                )
            )

        # ----------------------------------------------------
        # FP32 diagnostic forward
        #
        # This preserves the diagnostic interpretation and
        # gives us slots, token-normalized attention, and
        # raw P(slot | token) assignment.
        # ----------------------------------------------------

        with torch.inference_mode():

            projected_tokens = (
                model
                .extract_projected_tokens(
                    videos
                )
            )

            (
                slots,
                attn,
                assignment,
            ) = model.slot_attn(
                projected_tokens,
                initial_slots=initial_slots,
                return_assignment=True,
            )

            assign_stats = (
                assignment_statistics(
                    assignment
                )
            )

            repr_stats = (
                representation_attention_statistics(
                    slots,
                    attn,
                )
            )

        # ----------------------------------------------------
        # Official prediction path
        #
        # Separate pass is intentional:
        # prediction grouping should match evaluate().
        # ----------------------------------------------------

        logits = (
            _official_prediction_forward(
                model,
                videos,
                initial_slots,
                device,
            )
        )

        num_classes = (
            logits.shape[1]
        )

        if (
            max(topk_values)
            > num_classes
        ):
            raise ValueError(
                "Requested Top-k exceeds "
                f"number of classes: "
                f"k={max(topk_values)}, "
                f"classes={num_classes}."
            )

        pred = logits.argmax(
            dim=1
        )

        topk_indices = logits.topk(
            max(topk_values),
            dim=1,
            largest=True,
            sorted=True,
        ).indices

        # ----------------------------------------------------
        # One output row per example
        # ----------------------------------------------------

        for local_idx in range(
            batch_size
        ):

            dataset_idx = (
                offset
                + local_idx
            )

            metadata_row = (
                None
                if metadata is None
                else metadata[
                    dataset_idx
                ]
            )

            video_id = (
                _metadata_value(
                    metadata_row,
                    "video_id",
                    default=None,
                )
            )

            if video_id is None:
                video_id = (
                    _metadata_value(
                        metadata_row,
                        "id",
                        default=None,
                    )
                )

            row = {
                "dataset_index":
                    dataset_idx,

                "video_id":
                    video_id,

                "label":
                    int(
                        labels[
                            local_idx
                        ].item()
                    ),

                "pred":
                    int(
                        pred[
                            local_idx
                        ].item()
                    ),
            }

            original_label_id = (
                _metadata_value(
                    metadata_row,
                    "label_id",
                    default=None,
                )
            )

            if (
                original_label_id
                is not None
            ):
                row[
                    "original_label_id"
                ] = original_label_id

            # -----------------------------------------------
            # Per-example Top-k correctness
            # -----------------------------------------------

            for k in topk_values:

                is_correct = (
                    topk_indices[
                        local_idx,
                        :k,
                    ]
                    .eq(
                        labels[
                            local_idx
                        ]
                    )
                    .any()
                    .item()
                )

                row[
                    f"top{k}_correct"
                ] = bool(
                    is_correct
                )

            # -----------------------------------------------
            # Assignment-side scalar diagnostics
            # -----------------------------------------------

            assignment_scalar_keys = [
                "assign_entropy_mean",
                "assignment_confidence",
                "usage_entropy",
                "effective_slots",
                "soft_usage_min",
                "soft_usage_max",
                "hard_usage_min",
                "hard_usage_max",
                "hard_dead_slots",
            ]

            for key in (
                assignment_scalar_keys
            ):
                row[key] = float(
                    assign_stats[key][
                        local_idx
                    ].item()
                )

            # -----------------------------------------------
            # Representation/attention scalar diagnostics
            # -----------------------------------------------

            representation_scalar_keys = [
                "slot_cos_mean",
                "slot_cos_max",
                "attn_cos_mean",
                "attn_cos_max",
                "attn_js_mean",
                "slot_token_entropy_mean",
                "slot_token_entropy_min",
                "slot_token_entropy_max",
            ]

            for key in (
                representation_scalar_keys
            ):
                row[key] = float(
                    repr_stats[key][
                        local_idx
                    ].item()
                )

            # -----------------------------------------------
            # Per-slot usage columns
            # -----------------------------------------------

            soft_usage = (
                assign_stats[
                    "soft_usage"
                ][local_idx]
            )

            hard_usage = (
                assign_stats[
                    "hard_usage"
                ][local_idx]
            )

            for slot_idx in range(
                soft_usage.numel()
            ):

                row[
                    f"soft_usage_s{slot_idx}"
                ] = float(
                    soft_usage[
                        slot_idx
                    ].item()
                )

                row[
                    f"hard_usage_s{slot_idx}"
                ] = float(
                    hard_usage[
                        slot_idx
                    ].item()
                )

            row[
                "dominant_soft_slot"
            ] = int(
                soft_usage.argmax().item()
            )

            row[
                "dominant_hard_slot"
            ] = int(
                hard_usage.argmax().item()
            )

            rows.append(
                row
            )

        offset += batch_size

    if (
        offset
        != num_examples
    ):
        raise RuntimeError(
            f"Collected {offset} examples, "
            f"expected {num_examples}."
        )

    return rows


def summarize_diagnostic_rows(
    rows,
):
    """
    Compute simple dataset means from per-video diagnostic
    rows.

    Pairwise K=1 metrics are NaN-aware.
    """

    if not rows:
        raise ValueError(
            "rows must not be empty."
        )

    diagnostic_keys = [
        "assign_entropy_mean",
        "assignment_confidence",
        "usage_entropy",
        "effective_slots",
        "soft_usage_min",
        "soft_usage_max",
        "hard_usage_min",
        "hard_usage_max",
        "hard_dead_slots",
        "slot_cos_mean",
        "slot_cos_max",
        "attn_cos_mean",
        "attn_cos_max",
        "attn_js_mean",
        "slot_token_entropy_mean",
        "slot_token_entropy_min",
        "slot_token_entropy_max",
    ]

    summary = {
        "num_examples":
            len(rows),
    }

    for key in diagnostic_keys:

        values = torch.tensor(
            [
                float(row[key])
                for row in rows
            ],
            dtype=torch.float64,
        )

        finite = torch.isfinite(
            values
        )

        if finite.any():
            summary[key] = float(
                values[
                    finite
                ].mean().item()
            )
        else:
            summary[key] = float(
                "nan"
            )

    # ========================================================
    # Classification accuracy from the exact same rows
    # ========================================================

    topk_keys = sorted(
        key
        for key in rows[0]
        if (
            key.startswith("top")
            and key.endswith(
                "_correct"
            )
        )
    )

    for key in topk_keys:

        summary[
            key.replace(
                "_correct",
                "",
            )
        ] = sum(
            bool(
                row[key]
            )
            for row in rows
        ) / len(rows)

    return summary









def evaluate_dominant_slot_ablation(
    model,
    loader,
    device,
    *,
    slot_eval_mode="legacy_shared",
    slot_eval_seed=12345,
):
    """
    Compare the normal mean-of-slots representation against
    a representation containing only the dominant soft-usage
    slot.

    Dominant slot:
        argmax_k mean_n P(slot=k | token=n)

    The classifier is NOT retrained.

    Returns
    -------
    dict
        Aggregate metrics plus one row per example.
    """

    device = torch.device(
        device
    )

    valid_modes = {
        "legacy_shared",
        "per_video_deterministic",
    }

    if (
        slot_eval_mode
        not in valid_modes
    ):
        raise ValueError(
            "slot_eval_mode must be one of "
            f"{sorted(valid_modes)}."
        )

    if not hasattr(
        model,
        "slot_attn",
    ):
        raise ValueError(
            "model has no slot_attn module."
        )

    if not hasattr(
        model,
        "extract_projected_tokens",
    ):
        raise ValueError(
            "model must expose "
            "extract_projected_tokens()."
        )

    if not isinstance(
        loader.sampler,
        SequentialSampler,
    ):
        raise ValueError(
            "Dominant-slot ablation requires "
            "shuffle=False / SequentialSampler."
        )

    num_examples = len(
        loader.dataset
    )

    model.eval()

    # ========================================================
    # Corrected deterministic initialization bank
    # ========================================================

    noise_bank = None

    if (
        slot_eval_mode
        == "per_video_deterministic"
    ):
        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            slot_eval_seed
        )

        noise_bank = torch.randn(
            num_examples,
            model.slot_attn.num_slots,
            model.slot_attn.slot_dim,
            generator=generator,
        )

    rows = []

    offset = 0

    full_correct_total = 0
    dominant_correct_total = 0

    prediction_flip_total = 0

    full_correct_dominant_wrong = 0
    full_wrong_dominant_correct = 0

    dominant_slot_counts = torch.zeros(
        model.slot_attn.num_slots,
        dtype=torch.long,
    )

    # ========================================================
    # Evaluation
    # ========================================================

    for videos, labels in loader:

        batch_size = labels.size(0)

        videos = videos.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Initial slots
        # ----------------------------------------------------

        if noise_bank is None:

            initial_slots = None

        else:

            noise = noise_bank[
                offset:
                offset + batch_size
            ]

            initial_slots = (
                _build_per_video_initial_slots(
                    model,
                    noise,
                )
            )

        # ----------------------------------------------------
        # One internal forward.
        #
        # Keep the whole inference path under the same AMP
        # policy as official evaluation.
        # ----------------------------------------------------

        with (
            torch.inference_mode(),
            torch.cuda.amp.autocast(
                enabled=(
                    device.type == "cuda"
                )
            ),
        ):

            tokens = (
                model
                .extract_projected_tokens(
                    videos
                )
            )

            (
                slots,
                _attn,
                assignment,
            ) = model.slot_attn(
                tokens,
                initial_slots=initial_slots,
                return_assignment=True,
            )

            # -----------------------------------------------
            # Normal model representation
            # -----------------------------------------------

            full_repr = slots.mean(
                dim=1
            )

            full_logits = (
                model.classifier(
                    full_repr
                )
            )

            # -----------------------------------------------
            # Dominant slot according to SOFT usage
            # -----------------------------------------------

            soft_usage = (
                assignment.mean(
                    dim=-1
                )
            )

            dominant_idx = (
                soft_usage.argmax(
                    dim=1
                )
            )

            batch_indices = torch.arange(
                batch_size,
                device=slots.device,
            )

            dominant_repr = slots[
                batch_indices,
                dominant_idx,
            ]

            dominant_logits = (
                model.classifier(
                    dominant_repr
                )
            )

        # ====================================================
        # Predictions
        # ====================================================

        full_pred = (
            full_logits.argmax(
                dim=1
            )
        )

        dominant_pred = (
            dominant_logits.argmax(
                dim=1
            )
        )

        full_correct = full_pred.eq(
            labels
        )

        dominant_correct = (
            dominant_pred.eq(
                labels
            )
        )

        prediction_flip = (
            full_pred.ne(
                dominant_pred
            )
        )

        harmed = (
            full_correct
            & ~dominant_correct
        )

        helped = (
            ~full_correct
            & dominant_correct
        )

        # ====================================================
        # Aggregate counts
        # ====================================================

        full_correct_total += int(
            full_correct.sum().item()
        )

        dominant_correct_total += int(
            dominant_correct.sum().item()
        )

        prediction_flip_total += int(
            prediction_flip.sum().item()
        )

        full_correct_dominant_wrong += int(
            harmed.sum().item()
        )

        full_wrong_dominant_correct += int(
            helped.sum().item()
        )

        batch_counts = torch.bincount(
            dominant_idx.detach().cpu(),
            minlength=(
                model.slot_attn.num_slots
            ),
        )

        dominant_slot_counts += (
            batch_counts
        )

        # ====================================================
        # Per-example rows
        # ====================================================

        for i in range(
            batch_size
        ):

            rows.append({
                "dataset_index":
                    offset + i,

                "label":
                    int(
                        labels[i].item()
                    ),

                "full_pred":
                    int(
                        full_pred[i].item()
                    ),

                "dominant_pred":
                    int(
                        dominant_pred[i].item()
                    ),

                "full_correct":
                    bool(
                        full_correct[i].item()
                    ),

                "dominant_correct":
                    bool(
                        dominant_correct[
                            i
                        ].item()
                    ),

                "prediction_flip":
                    bool(
                        prediction_flip[
                            i
                        ].item()
                    ),

                "full_correct_dominant_wrong":
                    bool(
                        harmed[i].item()
                    ),

                "full_wrong_dominant_correct":
                    bool(
                        helped[i].item()
                    ),

                "dominant_slot":
                    int(
                        dominant_idx[
                            i
                        ].item()
                    ),

                "dominant_soft_usage":
                    float(
                        soft_usage[
                            i,
                            dominant_idx[i],
                        ].item()
                    ),
            })

        offset += batch_size

    if (
        offset
        != num_examples
    ):
        raise RuntimeError(
            f"Processed {offset} examples, "
            f"expected {num_examples}."
        )

    # ========================================================
    # Final summary
    # ========================================================

    full_top1 = (
        full_correct_total
        / num_examples
    )

    dominant_top1 = (
        dominant_correct_total
        / num_examples
    )

    prediction_flip_rate = (
        prediction_flip_total
        / num_examples
    )

    return {
        "num_examples":
            num_examples,

        "full_correct":
            full_correct_total,

        "dominant_correct":
            dominant_correct_total,

        "full_top1":
            full_top1,

        "dominant_top1":
            dominant_top1,

        "top1_drop":
            full_top1
            - dominant_top1,

        "prediction_flips":
            prediction_flip_total,

        "prediction_flip_rate":
            prediction_flip_rate,

        "full_correct_dominant_wrong":
            full_correct_dominant_wrong,

        "full_wrong_dominant_correct":
            full_wrong_dominant_correct,

        "dominant_slot_counts":
            dominant_slot_counts.tolist(),

        "rows":
            rows,
    }
    
    


def reshape_videomae_token_map(
    tensor,
    temporal_bins,
    grid_height,
    grid_width,
):
    """
    Reshape a VideoMAE token-axis tensor from:

        [B, K, N]

    to:

        [B, K, T, H, W]

    VideoMAE token order is T-major with spatial H,W
    positions inside each temporal tubelet.

    N must equal:

        T * H * W
    """

    if tensor.ndim != 3:
        raise ValueError(
            "tensor must have shape [B, K, N]. "
            f"Got {tuple(tensor.shape)}."
        )

    if temporal_bins < 1:
        raise ValueError(
            "temporal_bins must be >= 1."
        )

    if (
        grid_height < 1
        or grid_width < 1
    ):
        raise ValueError(
            "grid dimensions must be >= 1."
        )

    expected_tokens = (
        temporal_bins
        * grid_height
        * grid_width
    )

    actual_tokens = (
        tensor.shape[-1]
    )

    if (
        actual_tokens
        != expected_tokens
    ):
        raise ValueError(
            "Token count does not match "
            "the requested VideoMAE grid: "
            f"got N={actual_tokens}, "
            f"expected "
            f"{temporal_bins}x"
            f"{grid_height}x"
            f"{grid_width}"
            f"={expected_tokens}."
        )

    return tensor.reshape(
        tensor.shape[0],
        tensor.shape[1],
        temporal_bins,
        grid_height,
        grid_width,
    )


def videomae_temporal_statistics(
    assignment,
    attn,
    *,
    temporal_bins=8,
    grid_height=14,
    grid_width=14,
    eps=1e-8,
):
    """
    Quantify temporal Slot behavior for VideoMAE.

    Parameters
    ----------
    assignment:
        Raw P(slot | token), shape [B,K,N].

        Used to measure temporal slot competition.

    attn:
        Token-normalized per-slot attention,
        shape [B,K,N].

        Used to measure where each slot distributes its
        attention mass over temporal tubelets.

    temporal_bins:
        Number of VideoMAE temporal tubelets.

        Canonical setup:
            16 frames / tubelet_size 2 = 8

    grid_height, grid_width:
        Spatial VideoMAE token grid.

        Canonical setup:
            14 x 14

    Returns
    -------
    dict[str, torch.Tensor]

    Important outputs:

        assignment_temporal_usage:
            [B,K,T]

        attn_temporal_profile:
            [B,K,T]

        temporal_entropy:
            [B,K]

        temporal_cos_mean/max:
            [B]

        temporal_js_mean:
            [B]

        temporal_peak_mass:
            [B,K]

        temporal_peak_tubelet:
            [B,K]

        unique_peak_tubelets:
            [B]

        tubelet_assignment_confidence:
            [B,T]

        tubelet_effective_slots:
            [B,T]
    """

    # ========================================================
    # Validation
    # ========================================================

    if assignment.ndim != 3:
        raise ValueError(
            "assignment must have shape [B,K,N]."
        )

    if attn.ndim != 3:
        raise ValueError(
            "attn must have shape [B,K,N]."
        )

    if (
        assignment.shape
        != attn.shape
    ):
        raise ValueError(
            "assignment and attn must have "
            "the same shape."
        )

    batch_size, num_slots, _ = (
        assignment.shape
    )

    if not torch.isfinite(
        assignment
    ).all():
        raise ValueError(
            "assignment contains non-finite values."
        )

    if not torch.isfinite(
        attn
    ).all():
        raise ValueError(
            "attn contains non-finite values."
        )

    if (
        assignment < -1e-6
    ).any():
        raise ValueError(
            "assignment contains negative values."
        )

    if (
        attn < -1e-6
    ).any():
        raise ValueError(
            "attn contains negative values."
        )

    # P(slot | token) must sum over slots.
    if not torch.allclose(
        assignment.sum(dim=1),
        torch.ones_like(
            assignment[:, 0, :]
        ),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError(
            "assignment must sum to 1 "
            "over slots for every token."
        )

    # Token-normalized attention must sum over tokens.
    if not torch.allclose(
        attn.sum(dim=-1),
        torch.ones_like(
            attn[:, :, 0]
        ),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError(
            "attn must sum to 1 "
            "over tokens for every slot."
        )

    # ========================================================
    # Restore [T,H,W]
    # ========================================================

    assignment_grid = (
        reshape_videomae_token_map(
            assignment,
            temporal_bins,
            grid_height,
            grid_width,
        )
    )

    attn_grid = (
        reshape_videomae_token_map(
            attn,
            temporal_bins,
            grid_height,
            grid_width,
        )
    )

    # ========================================================
    # 1. Temporal assignment usage
    #
    # Average spatially within each tubelet:
    #
    #     [B,K,T,H,W]
    #       -> [B,K,T]
    #
    # For each t:
    #
    #     sum_K usage[...,t] = 1
    #
    # because assignment is P(slot | token).
    # ========================================================

    assignment_temporal_usage = (
        assignment_grid.mean(
            dim=(-1, -2)
        )
    )

    # ========================================================
    # 2. Per-slot temporal attention profile
    #
    # Sum spatial attention mass within each tubelet.
    #
    # Because attn is normalized across all N tokens:
    #
    #     sum_T profile = 1
    #
    # independently for every slot.
    # ========================================================

    attn_temporal_profile = (
        attn_grid.sum(
            dim=(-1, -2)
        )
    )

    # ========================================================
    # 3. Temporal entropy per slot
    #
    # How temporally spread is each slot?
    #
    # 0 -> all attention in one tubelet
    # 1 -> uniform over all T tubelets
    # ========================================================

    if temporal_bins == 1:

        temporal_entropy = (
            torch.zeros(
                batch_size,
                num_slots,
                dtype=attn.dtype,
                device=attn.device,
            )
        )

    else:

        profile_safe = (
            attn_temporal_profile
            .clamp_min(
                eps
            )
        )

        temporal_entropy = -(
            attn_temporal_profile
            * profile_safe.log()
        ).sum(
            dim=-1
        )

        temporal_entropy = (
            temporal_entropy
            / math.log(
                temporal_bins
            )
        )

    # ========================================================
    # 4. Pairwise similarity between slot temporal profiles
    # ========================================================

    profile_norm = F.normalize(
        attn_temporal_profile,
        p=2,
        dim=-1,
        eps=eps,
    )

    temporal_cos_matrix = (
        profile_norm
        @ profile_norm.transpose(
            1,
            2,
        )
    )

    temporal_cos_pairs = (
        _pairwise_upper_triangle(
            temporal_cos_matrix
        )
    )

    # ========================================================
    # 5. Pairwise temporal Jensen-Shannon divergence
    # ========================================================

    p = (
        attn_temporal_profile
        .unsqueeze(2)
    )

    q = (
        attn_temporal_profile
        .unsqueeze(1)
    )

    m = 0.5 * (
        p + q
    )

    p_safe = p.clamp_min(
        eps
    )

    q_safe = q.clamp_min(
        eps
    )

    m_safe = m.clamp_min(
        eps
    )

    kl_pm = (
        p
        * (
            p_safe.log()
            - m_safe.log()
        )
    ).sum(
        dim=-1
    )

    kl_qm = (
        q
        * (
            q_safe.log()
            - m_safe.log()
        )
    ).sum(
        dim=-1
    )

    temporal_js_matrix = (
        0.5
        * (
            kl_pm
            + kl_qm
        )
    )

    temporal_js_pairs = (
        _pairwise_upper_triangle(
            temporal_js_matrix
        )
        / math.log(2.0)
    )

    # ========================================================
    # 6. Temporal peak
    #
    # Which tubelet receives the largest attention mass
    # for every slot?
    # ========================================================

    temporal_peak_mass = (
        attn_temporal_profile.max(
            dim=-1
        ).values
    )

    temporal_peak_tubelet = (
        attn_temporal_profile.argmax(
            dim=-1
        )
    )

    # Number of distinct peak tubelets represented by slots.
    #
    # This is descriptive only:
    # different argmax tubelets do NOT automatically imply
    # meaningful temporal specialization.
    peak_one_hot = (
        F.one_hot(
            temporal_peak_tubelet,
            num_classes=temporal_bins,
        )
        .bool()
    )

    unique_peak_tubelets = (
        peak_one_hot
        .any(dim=1)
        .sum(dim=1)
    )

    # ========================================================
    # 7. Tubelet-wise assignment confidence
    #
    # Same definition as ordinary assignment confidence,
    # but kept separately for each temporal tubelet.
    #
    # First:
    #   max_K P(slot | token)
    #
    # then average H,W.
    # ========================================================

    tubelet_assignment_confidence = (
        assignment_grid
        .max(dim=1)
        .values
        .mean(
            dim=(-1, -2)
        )
    )

    # ========================================================
    # 8. Tubelet-wise effective number of slots
    #
    # Spatially averaged assignment at each t is a
    # distribution over K slots.
    #
    # effective = exp(entropy)
    # ========================================================

    temporal_usage_safe = (
        assignment_temporal_usage
        .clamp_min(
            eps
        )
    )

    tubelet_usage_entropy = -(
        assignment_temporal_usage
        * temporal_usage_safe.log()
    ).sum(
        dim=1
    )

    tubelet_effective_slots = (
        tubelet_usage_entropy.exp()
    )

    # ========================================================
    # K=1 pairwise temporal metrics
    # ========================================================

    if num_slots == 1:

        nan_values = torch.full(
            (batch_size,),
            float("nan"),
            dtype=attn.dtype,
            device=attn.device,
        )

        temporal_cos_mean = (
            nan_values.clone()
        )

        temporal_cos_max = (
            nan_values.clone()
        )

        temporal_js_mean = (
            nan_values.clone()
        )

    else:

        temporal_cos_mean = (
            temporal_cos_pairs.mean(
                dim=1
            )
        )

        temporal_cos_max = (
            temporal_cos_pairs.max(
                dim=1
            ).values
        )

        temporal_js_mean = (
            temporal_js_pairs.mean(
                dim=1
            )
        )

    # ========================================================
    # Output
    # ========================================================

    return {
        # Full maps for later visualization
        "assignment_grid":
            assignment_grid,

        "attn_grid":
            attn_grid,

        # Temporal curves
        "assignment_temporal_usage":
            assignment_temporal_usage,

        "attn_temporal_profile":
            attn_temporal_profile,

        # Temporal spread
        "temporal_entropy":
            temporal_entropy,

        "temporal_entropy_mean":
            temporal_entropy.mean(
                dim=1
            ),

        "temporal_entropy_min":
            temporal_entropy.min(
                dim=1
            ).values,

        "temporal_entropy_max":
            temporal_entropy.max(
                dim=1
            ).values,

        # Between-slot temporal similarity
        "temporal_cos_mean":
            temporal_cos_mean,

        "temporal_cos_max":
            temporal_cos_max,

        "temporal_js_mean":
            temporal_js_mean,

        # Temporal peaks
        "temporal_peak_mass":
            temporal_peak_mass,

        "temporal_peak_mass_mean":
            temporal_peak_mass.mean(
                dim=1
            ),

        "temporal_peak_mass_max":
            temporal_peak_mass.max(
                dim=1
            ).values,

        "temporal_peak_tubelet":
            temporal_peak_tubelet,

        "unique_peak_tubelets":
            unique_peak_tubelets,

        # Tubelet-wise slot competition
        "tubelet_assignment_confidence":
            tubelet_assignment_confidence,

        "tubelet_assignment_confidence_mean":
            tubelet_assignment_confidence.mean(
                dim=1
            ),

        "tubelet_assignment_confidence_min":
            tubelet_assignment_confidence.min(
                dim=1
            ).values,

        "tubelet_assignment_confidence_max":
            tubelet_assignment_confidence.max(
                dim=1
            ).values,

        "tubelet_effective_slots":
            tubelet_effective_slots,

        "tubelet_effective_slots_mean":
            tubelet_effective_slots.mean(
                dim=1
            ),

        "tubelet_effective_slots_min":
            tubelet_effective_slots.min(
                dim=1
            ).values,

        "tubelet_effective_slots_max":
            tubelet_effective_slots.max(
                dim=1
            ).values,
    }