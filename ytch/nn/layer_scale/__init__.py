from ytch.nn.elementwise_affine import ElementwiseAffine


class LayerScale(ElementwiseAffine):
    """LayerScale: y = scale * x with small initialization."""

    def __init__(self, dim: int, init_values: float = 1e-6, device=None, dtype=None):
        super().__init__(
            dim=dim,
            scale=init_values,
            bias=False,
            device=device,
            dtype=dtype,
        )
