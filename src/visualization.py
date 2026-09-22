import math

import torch


def _to_cpu_tensor(
    x,
    *,
    name="tensor",
):
    """
    Convert a Tensor-like input to a detached CPU tensor.
    """

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if not torch.isfinite(x).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return (
        x
        .detach()
        .cpu()
    )


def select_examples_by_metric(
    rows,
    metric,
    *,
    correct=None,
):
    """
    Objectively choose low / median / high examples for a
    diagnostic metric.

    Parameters
    ----------
    rows:
        Per-video rows, e.g. from collect_slot_diagnostics().

    metric:
        Numeric field to rank by.

    correct:
        Optional correctness filter.

        None:
            use every row

        True:
            only correct examples

        False:
            only wrong examples

        Uses "top1_correct" when filtering.

    Returns
    -------
    dict
        {
            "low": row,
            "median": row,
            "high": row,
        }
    """

    if not rows:
        raise ValueError(
            "rows must not be empty."
        )

    candidates = []

    for row in rows:

        if metric not in row:
            raise KeyError(
                f"Metric {metric!r} is missing "
                "from at least one row."
            )

        if correct is not None:

            if "top1_correct" not in row:
                raise KeyError(
                    "correct filtering requires "
                    "'top1_correct' in each row."
                )

            if (
                bool(row["top1_correct"])
                != bool(correct)
            ):
                continue

        value = float(
            row[metric]
        )

        if math.isfinite(value):
            candidates.append(
                row
            )

    if not candidates:
        raise ValueError(
            "No finite rows remain after filtering."
        )

    candidates = sorted(
        candidates,
        key=lambda row:
            float(row[metric]),
    )

    median_idx = (
        len(candidates)
        // 2
    )

    return {
        "low":
            candidates[0],

        "median":
            candidates[
                median_idx
            ],

        "high":
            candidates[-1],
    }


def winner_confidence_maps(
    assignment_grid,
):
    """
    Compute winning-slot and confidence maps from raw
    P(slot | token).

    Parameters
    ----------
    assignment_grid:
        Shape:
            [K,T,H,W]

        Probabilities must sum to 1 over K.

    Returns
    -------
    winner:
        [T,H,W], integer slot index.

    confidence:
        [T,H,W], winning probability in [0,1].
    """

    assignment_grid = _to_cpu_tensor(
        assignment_grid,
        name="assignment_grid",
    )

    if assignment_grid.ndim != 4:
        raise ValueError(
            "assignment_grid must have shape "
            "[K,T,H,W]."
        )

    if (
        assignment_grid < -1e-6
    ).any():
        raise ValueError(
            "assignment_grid contains "
            "negative probabilities."
        )

    sums = assignment_grid.sum(
        dim=0
    )

    if not torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError(
            "assignment_grid must sum to 1 "
            "over slots at every token."
        )

    confidence, winner = (
        assignment_grid.max(
            dim=0
        )
    )

    return (
        winner,
        confidence,
    )


def plot_slot_assignment_maps(
    assignment_grid,
    *,
    temporal_bin,
    title=None,
):
    """
    Plot raw P(slot | token) spatial maps for one temporal
    tubelet.

    CRITICAL:
        Every slot uses the SAME fixed visualization scale:

            vmin = 0
            vmax = 1

        No per-slot min-max normalization is performed.

    Parameters
    ----------
    assignment_grid:
        [K,T,H,W]

    temporal_bin:
        Tubelet index.

    Returns
    -------
    matplotlib Figure
    """

    import matplotlib.pyplot as plt

    assignment_grid = _to_cpu_tensor(
        assignment_grid,
        name="assignment_grid",
    )

    if assignment_grid.ndim != 4:
        raise ValueError(
            "assignment_grid must have shape "
            "[K,T,H,W]."
        )

    num_slots, num_times, _, _ = (
        assignment_grid.shape
    )

    if not (
        0
        <= temporal_bin
        < num_times
    ):
        raise IndexError(
            f"temporal_bin={temporal_bin} "
            f"is outside [0,{num_times - 1}]."
        )

    cols = min(
        num_slots,
        4,
    )

    rows = math.ceil(
        num_slots / cols
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            4 * cols,
            4 * rows,
        ),
        squeeze=False,
    )

    last_image = None

    for slot_idx in range(
        num_slots
    ):

        row_idx = (
            slot_idx // cols
        )

        col_idx = (
            slot_idx % cols
        )

        ax = axes[
            row_idx,
            col_idx,
        ]

        spatial_map = (
            assignment_grid[
                slot_idx,
                temporal_bin,
            ]
        )

        last_image = ax.imshow(
            spatial_map.numpy(),
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title(
            f"Slot {slot_idx}"
        )

        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused panels.
    for slot_idx in range(
        num_slots,
        rows * cols,
    ):
        axes[
            slot_idx // cols,
            slot_idx % cols,
        ].axis("off")

    if title is None:
        title = (
            "Raw P(slot | token) — "
            f"tubelet {temporal_bin}"
        )

    fig.suptitle(
        title
    )

    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.02,
            label="P(slot | token)",
        )

    return fig


def plot_winner_confidence_maps(
    assignment_grid,
    *,
    temporal_bin,
    title=None,
):
    """
    Plot:
        1. winning slot at each spatial token
        2. corresponding winning probability

    Confidence always uses fixed [0,1] scaling.
    """

    import matplotlib.pyplot as plt

    winner, confidence = (
        winner_confidence_maps(
            assignment_grid
        )
    )

    num_times = winner.shape[0]

    if not (
        0
        <= temporal_bin
        < num_times
    ):
        raise IndexError(
            f"temporal_bin={temporal_bin} "
            f"is outside [0,{num_times - 1}]."
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9, 4),
    )

    winner_image = axes[0].imshow(
        winner[
            temporal_bin
        ].numpy(),
        interpolation="nearest",
    )

    axes[0].set_title(
        "Winning slot"
    )

    axes[0].set_xticks([])
    axes[0].set_yticks([])

    fig.colorbar(
        winner_image,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
    )

    confidence_image = (
        axes[1].imshow(
            confidence[
                temporal_bin
            ].numpy(),
            vmin=0.0,
            vmax=1.0,
        )
    )

    axes[1].set_title(
        "Winning-slot confidence"
    )

    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.colorbar(
        confidence_image,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="max P(slot | token)",
    )

    if title is None:
        title = (
            f"Tubelet {temporal_bin}"
        )

    fig.suptitle(
        title
    )

    fig.tight_layout()

    return fig


def plot_temporal_profiles(
    attn_temporal_profile,
    *,
    title=None,
):
    """
    Plot each slot's attention mass over temporal tubelets.

    Parameters
    ----------
    attn_temporal_profile:
        [K,T]

        Each slot should sum to approximately 1 over T.

    Returns
    -------
    matplotlib Figure
    """

    import matplotlib.pyplot as plt

    profile = _to_cpu_tensor(
        attn_temporal_profile,
        name="attn_temporal_profile",
    )

    if profile.ndim != 2:
        raise ValueError(
            "attn_temporal_profile must have "
            "shape [K,T]."
        )

    if (
        profile < -1e-6
    ).any():
        raise ValueError(
            "Temporal profile contains "
            "negative values."
        )

    sums = profile.sum(
        dim=-1
    )

    if not torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError(
            "Each temporal profile must "
            "sum to 1 over tubelets."
        )

    num_slots, num_times = (
        profile.shape
    )

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    x = torch.arange(
        num_times
    ).numpy()

    for slot_idx in range(
        num_slots
    ):
        ax.plot(
            x,
            profile[
                slot_idx
            ].numpy(),
            marker="o",
            label=f"Slot {slot_idx}",
        )

    ax.set_xlabel(
        "Temporal tubelet"
    )

    ax.set_ylabel(
        "Attention mass"
    )

    ax.set_xticks(
        range(num_times)
    )

    # Every slot distribution lies in [0,1].
    ax.set_ylim(
        0.0,
        1.0,
    )

    if title is None:
        title = (
            "Per-slot temporal attention"
        )

    ax.set_title(
        title
    )

    ax.legend()

    fig.tight_layout()

    return fig


def plot_slot_tubelet_heatmap(
    attn_temporal_profile,
    *,
    title=None,
):
    """
    Display the same temporal profiles as a K x T heatmap.

    Uses a fixed [0,1] scale.
    """

    import matplotlib.pyplot as plt

    profile = _to_cpu_tensor(
        attn_temporal_profile,
        name="attn_temporal_profile",
    )

    if profile.ndim != 2:
        raise ValueError(
            "attn_temporal_profile must have "
            "shape [K,T]."
        )

    num_slots, num_times = (
        profile.shape
    )

    fig, ax = plt.subplots(
        figsize=(
            max(
                7,
                num_times,
            ),
            max(
                3,
                0.75 * num_slots,
            ),
        )
    )

    image = ax.imshow(
        profile.numpy(),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xlabel(
        "Temporal tubelet"
    )

    ax.set_ylabel(
        "Slot"
    )

    ax.set_xticks(
        range(num_times)
    )

    ax.set_yticks(
        range(num_slots)
    )

    if title is None:
        title = (
            "Slot × tubelet attention mass"
        )

    ax.set_title(
        title
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Attention mass",
    )

    fig.tight_layout()

    return fig


def plot_video_frames(
    frames,
    *,
    frame_indices=None,
    max_frames=8,
    title=None,
):
    """
    Plot raw RGB video frames.

    Parameters
    ----------
    frames:
        [T,H,W,3]

        Expected to be display-ready RGB values.

        Do NOT pass normalized model tensors unless they
        have first been converted back to display RGB.

    frame_indices:
        Optional subset of frame indices.

    max_frames:
        If frame_indices is None, uniformly select at most
        this many frames.

    Returns
    -------
    matplotlib Figure
    """

    import matplotlib.pyplot as plt

    frames = _to_cpu_tensor(
        frames,
        name="frames",
    )

    if (
        frames.ndim != 4
        or frames.shape[-1] != 3
    ):
        raise ValueError(
            "frames must have shape "
            "[T,H,W,3]."
        )

    num_frames = (
        frames.shape[0]
    )

    if frame_indices is None:

        count = min(
            num_frames,
            max_frames,
        )

        if count == 1:
            frame_indices = [0]

        else:
            frame_indices = (
                torch.linspace(
                    0,
                    num_frames - 1,
                    count,
                )
                .long()
                .tolist()
            )

    else:
        frame_indices = [
            int(i)
            for i in frame_indices
        ]

    for idx in frame_indices:

        if not (
            0
            <= idx
            < num_frames
        ):
            raise IndexError(
                f"frame index {idx} "
                "is outside video."
            )

    count = len(
        frame_indices
    )

    cols = min(
        count,
        4,
    )

    rows = math.ceil(
        count / cols
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            4 * cols,
            3 * rows,
        ),
        squeeze=False,
    )

    for plot_idx, frame_idx in enumerate(
        frame_indices
    ):

        ax = axes[
            plot_idx // cols,
            plot_idx % cols,
        ]

        frame = frames[
            frame_idx
        ].numpy()

        ax.imshow(
            frame
        )

        ax.set_title(
            f"Frame {frame_idx}"
        )

        ax.set_xticks([])
        ax.set_yticks([])

    for plot_idx in range(
        count,
        rows * cols,
    ):
        axes[
            plot_idx // cols,
            plot_idx % cols,
        ].axis("off")

    if title is not None:
        fig.suptitle(
            title
        )

    fig.tight_layout()

    return fig