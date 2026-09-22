import torch


def choose_topk(
    num_classes,
):
    """
    Default Top-k metrics used by the canonical experiments.

    5 classes or fewer:
        Top-1

    6-20 classes:
        Top-1, Top-3

    More than 20 classes:
        Top-1, Top-5
    """

    if num_classes <= 5:
        return [1]

    if num_classes <= 20:
        return [1, 3]

    return [1, 5]


def normalize_topk(
    topk,
    num_classes,
):
    """
    Normalize an explicitly requested Top-k specification.

    Top-1 is always included.

    Examples
    --------
    normalize_topk(None, 20)
        -> [1, 3]

    normalize_topk(5, 50)
        -> [1, 5]

    normalize_topk([1, 3, 5], 50)
        -> [1, 3, 5]
    """

    if topk is None:
        return choose_topk(
            num_classes
        )

    if isinstance(
        topk,
        int,
    ):
        ks = {
            1,
            int(topk),
        }

    else:
        ks = {
            1,
            *[
                int(k)
                for k in topk
            ],
        }

    return sorted(
        k
        for k in ks
        if 1 <= k <= num_classes
    )


def topk_correct_counts(
    logits,
    labels,
    topk_values,
):
    """
    Return the number of correctly classified examples
    for every requested value of k.

    Parameters
    ----------
    logits:
        [B, C]

    labels:
        [B]

    topk_values:
        Iterable such as [1], [1, 3], or [1, 3, 5].

    Returns
    -------
    dict
        Example:
            {
                1: 7,
                3: 9,
                5: 10,
            }
    """

    max_k = max(
        topk_values
    )

    _, pred = logits.topk(
        max_k,
        dim=1,
        largest=True,
        sorted=True,
    )

    pred = pred.t()

    correct = pred.eq(
        labels.view(
            1,
            -1,
        )
    )

    out = {}

    for k in topk_values:
        out[k] = int(
            correct[
                :k
            ]
            .any(
                dim=0
            )
            .sum()
            .item()
        )

    return out



def evaluate(
    model,
    loader,
    criterion,
    topk_values,
    device,
):
    """
    Standard classifier evaluation.

    Returns
    -------
    avg_loss : float

    metrics : dict
        Example:
            {
                "top1": 0.52,
                "top3": 0.78,
            }
    """

    device = torch.device(
        device
    )

    model.eval()

    total = 0
    total_loss = 0.0

    topk_totals = {
        k: 0
        for k in topk_values
    }

    with torch.no_grad():

        for videos, labels in loader:

            videos = videos.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            with torch.cuda.amp.autocast(
                enabled=(
                    device.type == "cuda"
                )
            ):
                outputs = model(
                    videos
                )

                loss = criterion(
                    outputs,
                    labels,
                )

            batch_size = (
                labels.size(0)
            )

            total += batch_size

            total_loss += (
                loss.item()
                * batch_size
            )

            counts = (
                topk_correct_counts(
                    outputs,
                    labels,
                    topk_values,
                )
            )

            for k, value in (
                counts.items()
            ):
                topk_totals[k] += value

    metrics = {
        f"top{k}":
            topk_totals[k]
            / max(1, total)

        for k in topk_values
    }

    avg_loss = (
        total_loss
        / max(1, total)
    )

    return (
        avg_loss,
        metrics,
    )
    
    
    
    
    
def evaluate_slot_per_video_init(
    model,
    loader,
    criterion,
    topk_values,
    device,
    init_seed=12345,
):
    """
    Deterministic Slot evaluation with one independent
    initialization per dataset example.

    Dataset row i always receives the same independently
    sampled initial slots, regardless of DataLoader batch size.

    IMPORTANT:
        The loader must iterate the dataset in deterministic
        dataset order, i.e. shuffle=False.

    This is the corrected Slot evaluation protocol used by
    the canonical experiments.
    """

    device = torch.device(
        device
    )

    model.eval()

    if not hasattr(
        model,
        "slot_attn",
    ):
        raise ValueError(
            "Model has no slot_attn module."
        )

    num_examples = len(
        loader.dataset
    )

    num_slots = (
        model.slot_attn.num_slots
    )

    slot_dim = (
        model.slot_attn.slot_dim
    )

    # --------------------------------------------------------
    # Generate the ENTIRE deterministic initialization bank
    # before iterating over batches.
    #
    # This makes initialization independent of batch size.
    # Dataset example i always receives noise_bank[i].
    # --------------------------------------------------------

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        init_seed
    )

    noise_bank = torch.randn(
        num_examples,
        num_slots,
        slot_dim,
        generator=generator,
    )

    total = 0
    total_loss = 0.0

    topk_totals = {
        k: 0
        for k in topk_values
    }

    offset = 0

    for videos, labels in loader:

        batch_size = videos.size(0)

        videos = videos.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Retrieve the deterministic noise assigned to the
        # exact dataset rows in this batch.
        # ----------------------------------------------------

        noise = noise_bank[
            offset:
            offset + batch_size
        ].to(
            device=device,
            dtype=(
                model
                .slot_attn
                .slots_mu
                .dtype
            ),
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

        initial_slots = (
            mu
            + sigma * noise
        )

        with (
            torch.inference_mode(),
            torch.cuda.amp.autocast(
                enabled=(
                    device.type == "cuda"
                )
            ),
        ):
            outputs = model(
                videos,
                initial_slots=initial_slots,
            )

            loss = criterion(
                outputs,
                labels,
            )

        total += batch_size

        total_loss += (
            loss.item()
            * batch_size
        )

        counts = topk_correct_counts(
            outputs,
            labels,
            topk_values,
        )

        for k, value in (
            counts.items()
        ):
            topk_totals[k] += value

        offset += batch_size

    if offset != num_examples:
        raise RuntimeError(
            f"Evaluated {offset} examples, "
            f"expected {num_examples}."
        )

    metrics = {
        f"top{k}": (
            topk_totals[k]
            / max(1, total)
        )
        for k in topk_values
    }

    avg_loss = (
        total_loss
        / max(1, total)
    )

    return (
        avg_loss,
        metrics,
    )
    
    
