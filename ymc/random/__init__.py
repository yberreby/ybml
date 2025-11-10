import numpy as np


def sample_by_tail_ratio(
    a: int, b: int, tail_ratio: float, size: int | tuple[int, ...] = ()
) -> np.ndarray:
    """Sample from {a..b} where P(b)/P(a) = tail_ratio ∈ [0,1].

    tail_ratio=0 → all mass at a; tail_ratio=1 → uniform.
    Creates geometric distribution via p_i = q^i where q = tail_ratio^(1/(n-1)).
    """
    assert 0 <= tail_ratio <= 1, f"tail_ratio must be in [0, 1], got {tail_ratio=}"
    assert a <= b, f"a must be <= b, got {a=}, {b=}"

    if tail_ratio == 0:
        return np.full(size, a, dtype=int)
    if tail_ratio == 1:
        return np.random.randint(a, b + 1, size=size)
    if a == b:
        return np.full(size, a, dtype=int)

    vals = np.arange(a, b + 1)
    q = float(tail_ratio) ** (1.0 / (len(vals) - 1))
    p = q ** (vals - a)
    p /= p.sum()
    return np.random.choice(vals, size=size, p=p)
