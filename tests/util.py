import torch

MAX_ABS_DIFF_THRESHOLD = 0.25
MEAN_ABS_DIFF_THRESHOLD = 0.1
TIGHT_MAX_ABS_DIFF_THRESHOLD = 1e-4
TIGHT_MEAN_ABS_DIFF_THRESHOLD = 1e-5


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape
    return (a.float() - b.float()).abs().max().item()


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape
    return (a.float() - b.float()).abs().mean().item()


def print_diff(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape == b.shape:
        print(
            f"{name}: max_abs={max_abs_diff(a, b):.6f}, mean_abs={mean_abs_diff(a, b):.6f}, shape={a.shape}"
        )
    else:
        print(f"{name}: shape mismatch! {a.shape} != {b.shape}")


def assert_allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    max_diff_tolerance,
    mean_diff_tolerance,
):
    assert a.shape == b.shape
    assert max_abs_diff(a, b) < max_diff_tolerance
    assert mean_abs_diff(a, b) < mean_diff_tolerance


def assert_allclose_loose(a: torch.Tensor, b: torch.Tensor):
    assert_allclose(
        a,
        b,
        max_diff_tolerance=MAX_ABS_DIFF_THRESHOLD,
        mean_diff_tolerance=MEAN_ABS_DIFF_THRESHOLD,
    )


def assert_allclose_tight(a: torch.Tensor, b: torch.Tensor):
    assert_allclose(
        a,
        b,
        max_diff_tolerance=TIGHT_MAX_ABS_DIFF_THRESHOLD,
        mean_diff_tolerance=TIGHT_MEAN_ABS_DIFF_THRESHOLD,
    )
