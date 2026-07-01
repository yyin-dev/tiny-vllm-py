import torch


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape
    return (a.float() - b.float()).abs().max().item()


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape
    return (a.float() - b.float()).abs().mean().item()
