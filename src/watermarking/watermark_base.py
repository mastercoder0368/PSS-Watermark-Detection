"""
Watermark Base Classes
Base implementation for watermarking functionality.
"""

import torch
from typing import Optional, List


class WatermarkBase:
    """Base class for watermarking operations."""

    def __init__(
            self,
            vocab: List[int] = None,
            gamma: float = 0.25,
            delta: float = 1.5,
            seeding_scheme: str = "simple_1",
            hash_key: int = 15485863,
            select_green_tokens: bool = True,
    ):
        """
        Initialize watermark base.

        Args:
            vocab: Vocabulary indices
            gamma: Proportion of green tokens
            delta: Green list bias
            seeding_scheme: Seeding scheme for RNG
            hash_key: Hash key for seeding
            select_green_tokens: Whether to select green tokens directly
        """
        self.vocab = vocab
        self.vocab_size = len(vocab) if vocab else 0
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens

    def _seed_rng(self, input_ids: torch.Tensor,
                  seeding_scheme: Optional[str] = None) -> None:
        """
        Seed the random number generator based on input.

        Args:
            input_ids: Input token IDs
            seeding_scheme: Override seeding scheme
        """
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            # Use last token as seed
            assert input_ids.shape[-1] >= 1, \
                f"seeding_scheme={seeding_scheme} requires at least 1 token"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unknown seeding_scheme: {seeding_scheme}")

    def _get_greenlist_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate the green list for given input.

        Args:
            input_ids: Input token IDs

        Returns:
            Tensor of green token IDs
        """
        self._seed_rng(input_ids)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size,
            device=input_ids.device,
            generator=self.rng
        )

        if self.select_green_tokens:
            # Select first greenlist_size tokens
            greenlist_ids = vocab_permutation[:greenlist_size]
        else:
            # Select from red tokens (last greenlist_size)
            greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size):]

        return greenlist_ids
