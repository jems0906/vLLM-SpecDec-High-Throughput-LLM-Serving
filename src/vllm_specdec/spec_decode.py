from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class ProposalWindow:
    draft_tokens: list[int]
    target_tokens: list[int]
    accepted_tokens: list[int]
    rejected_token: int | None
    acceptance_rate: float


def accept_tokens(draft_tokens: Sequence[int], target_tokens: Sequence[int]) -> ProposalWindow:
    accepted: list[int] = []
    rejected_token: int | None = None

    for draft_token, target_token in zip(draft_tokens, target_tokens):
        if draft_token == target_token:
            accepted.append(draft_token)
            continue
        rejected_token = target_token
        break

    if rejected_token is None and len(target_tokens) > len(accepted):
        rejected_token = target_tokens[len(accepted)]

    acceptance_rate = len(accepted) / max(1, len(draft_tokens))
    return ProposalWindow(
        draft_tokens=list(draft_tokens),
        target_tokens=list(target_tokens),
        accepted_tokens=accepted,
        rejected_token=rejected_token,
        acceptance_rate=acceptance_rate,
    )


class SpeculativeDecoder:
    def __init__(self, num_speculative_tokens: int = 5) -> None:
        if num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        self.num_speculative_tokens = num_speculative_tokens

    def verify(self, draft_tokens: Sequence[int], target_tokens: Sequence[int]) -> ProposalWindow:
        return accept_tokens(draft_tokens[: self.num_speculative_tokens], target_tokens)

    def merge_step(self, draft_tokens: Sequence[int], target_tokens: Sequence[int]) -> list[int]:
        window = self.verify(draft_tokens, target_tokens)
        merged = list(window.accepted_tokens)
        if window.rejected_token is not None:
            merged.append(window.rejected_token)
        return merged
