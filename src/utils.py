import torch

import math


from typing import Callable


def rel_bucket(rel_pos: torch.Tensor, decoder: bool, buckets=32, max_dist=128) -> torch.Tensor:
    if decoder:
        rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))
    else:
        half = buckets // 2
        rel_pos = torch.abs(rel_pos)
    exact = buckets // 2
    small = rel_pos < exact
    large = exact + (
        torch.log(rel_pos.float() / exact) / math.log(max_dist / exact) * (buckets - exact)
    ).long()
    large = torch.min(large, torch.full_like(large, buckets - 1))
    return torch.where(small, rel_pos, large)



def bias(q_len: int, k_len: int, embed: Callable, decoder: bool, device=None) -> torch.Tensor:
    device = torch.device('cpu')
    q_pos = torch.arange(q_len, dtype=torch.long, device=device).view(-1, 1)
    k_pos = torch.arange(k_len, dtype=torch.long, device=device).view(1, -1)
    rel = k_pos - q_pos
    buckets = rel_bucket(rel, decoder)
    vals = embed(buckets).permute(2, 0, 1).unsqueeze(0)
    return vals



