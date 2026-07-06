import torch


def append_kvs(
    kvs: list[tuple[torch.Tensor, torch.Tensor]],
    new_kvs: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    updated = []
    for (k_prefix, v_prefix), (new_k, new_v) in zip(kvs, new_kvs, strict=True):
        updated.append(
            (
                torch.cat([k_prefix, new_k], dim=-2),
                torch.cat([v_prefix, new_v], dim=-2),
            )
        )
    return updated
