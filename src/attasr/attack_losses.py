from typing import Callable, Protocol

import torch


def linf_norm(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm((target - x), ord=float("inf"), dim=1).sum(dim=-1)


def l2_norm(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum((target - x) ** 2, dim=-1, keepdim=True).sum()


class AttackLossProtocol(Protocol):
    def __call__(
        self,
        batch_logits: torch.Tensor,
        adv_x: torch.Tensor,
        input_feats: torch.Tensor,
    ) -> torch.Tensor:
        ...


def get_attack_loss(
    skip_tokens: set[int],
    eos_token_id: int,
    adv_dist_factor: float,
    adv_dist_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    minimize_eos_all_pos: bool = False,
) -> AttackLossProtocol:
    def _attack_loss(
        batch_logits: torch.Tensor,
        adv_x: torch.Tensor,
        input_feats: torch.Tensor,
    ) -> torch.Tensor:
        logits = batch_logits[0]
        sfmax = torch.softmax(logits[-1], dim=0)
        tokens_sorted = torch.argsort(sfmax)
        tail = -2
        while tokens_sorted[tail].item() in skip_tokens:
            tail -= 1

        if minimize_eos_all_pos:
            eos_penalty = logits[:, eos_token_id].sum()
        else:
            eos_penalty = logits[-1][eos_token_id]

        l1 = eos_penalty - logits[-1][tokens_sorted[tail]]
        l2 = adv_dist_criterion(adv_x, input_feats)
        return l1 + adv_dist_factor * l2

    return _attack_loss
