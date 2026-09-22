def get_checkpoint_model_state(
    model,
    model_name,
):
    """
    Return the model state that should be stored in a
    checkpoint.

    Frozen pretrained transformer backbones are excluded
    so that every checkpoint does not duplicate hundreds
    of MB of unchanged weights.

    Trainable video-ResNet backbones are stored in full.
    """

    full_state = model.state_dict()

    # --------------------------------------------------------
    # ViT
    #
    # Canonical ViT models expose the frozen backbone as:
    #     self.vit
    # --------------------------------------------------------

    if model_name.startswith("vit"):
        return {
            key: value.detach().cpu()
            for key, value in full_state.items()
            if not key.startswith("vit.")
        }

    # --------------------------------------------------------
    # TimeSformer
    #
    # Canonical TimeSformer models expose the frozen
    # backbone as:
    #     self.backbone
    # --------------------------------------------------------

    if model_name.startswith(
        "timesformer"
    ):
        return {
            key: value.detach().cpu()
            for key, value in full_state.items()
            if not key.startswith(
                "backbone."
            )
        }

    # --------------------------------------------------------
    # VideoMAE
    #
    # Canonical VideoMAE models also expose the frozen
    # backbone as:
    #     self.backbone
    # --------------------------------------------------------

    if model_name.startswith(
        "videomae"
    ):
        return {
            key: value.detach().cpu()
            for key, value in full_state.items()
            if not key.startswith(
                "backbone."
            )
        }

    # --------------------------------------------------------
    # R3D / MC3
    #
    # Their backbones are trainable, so preserve the entire
    # state, including BatchNorm parameters and buffers.
    # --------------------------------------------------------

    return {
        key: value.detach().cpu()
        for key, value in full_state.items()
    }


def load_checkpoint_model_state(
    model,
    state,
):
    """
    Load one of our compact checkpoint model states.

    strict=False is intentional because frozen pretrained
    transformer backbone weights are not stored in the
    checkpoint.

    Missing keys are therefore expected for those models.
    Unexpected keys are not expected and are treated as an
    error.

    Returns
    -------
    missing : list[str]
        Keys present in the instantiated model but absent
        from the compact checkpoint.
    """

    missing, unexpected = (
        model.load_state_dict(
            state,
            strict=False,
        )
    )

    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint keys: "
            f"{unexpected}"
        )

    return missing