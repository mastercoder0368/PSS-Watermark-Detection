"""
Watermark Detector
Detects watermarks in generated texts.
"""

import math
import torch
import scipy.stats
from typing import Dict, Optional
from collections import Counter
from nltk.util import ngrams

from .watermark_base import WatermarkBase


class WatermarkDetector(WatermarkBase):
    """Detector for watermarks in text."""

    def __init__(
            self,
            *args,
            device: str = None,
            tokenizer=None,
            z_threshold: float = 4.0,
            ignore_repeated_bigrams: bool = False,
            **kwargs
    ):
        """
        Initialize watermark detector.

        Args:
            device: Device to use
            tokenizer: Tokenizer instance
            z_threshold: Z-score threshold for detection
            ignore_repeated_bigrams: Whether to ignore repeated bigrams
        """
        super().__init__(*args, **kwargs)

        assert device, "Must provide device"
        assert tokenizer, "Must provide tokenizer"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unknown seeding_scheme: {self.seeding_scheme}")

        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", \
                "Repeated bigram variant requires simple_1 seeding"

    def _compute_z_score(self, observed_count: int, T: int) -> float:
        """
        Compute z-score for detection.

        Args:
            observed_count: Number of green tokens observed
            T: Total tokens scored

        Returns:
            Z-score
        """
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = math.sqrt(T * expected_count * (1 - expected_count))

        if denom == 0:
            return 0.0

        return numer / denom

    def _compute_p_value(self, z: float) -> float:
        """
        Compute p-value from z-score.

        Args:
            z: Z-score

        Returns:
            P-value
        """
        return scipy.stats.norm.sf(z)

    def _score_sequence(self, input_ids: torch.Tensor, **kwargs) -> Dict:
        """
        Score a tokenized sequence for watermark presence.

        Args:
            input_ids: Token IDs

        Returns:
            Dictionary with scoring results
        """
        return_green_token_mask = kwargs.get("return_green_token_mask", False)

        if self.ignore_repeated_bigrams:
            # Count unique bigrams only once
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())

            for bigram in freq.keys():
                prefix = torch.tensor([bigram[0]], device=self.device)
                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = bigram[1] in greenlist_ids

            green_token_count = sum(bigram_table.values())
            green_token_mask = None
        else:
            # Standard scoring
            num_tokens_scored = len(input_ids) - self.min_prefix_len

            if num_tokens_scored < 1:
                raise ValueError(
                    f"Must have at least 1 token after prefix of {self.min_prefix_len}"
                )

            green_token_count = 0
            green_token_mask = []

            for idx in range(self.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                greenlist_ids = self._get_greenlist_ids(input_ids[:idx])

                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)

        # Calculate scores
        score_dict = {
            'num_tokens_scored': num_tokens_scored,
            'num_green_tokens': green_token_count,
            'green_fraction': green_token_count / num_tokens_scored if num_tokens_scored > 0 else 0,
            'z_score': self._compute_z_score(green_token_count, num_tokens_scored),
        }

        score_dict['p_value'] = self._compute_p_value(score_dict['z_score'])

        if green_token_mask is not None and return_green_token_mask:
            score_dict['green_token_mask'] = green_token_mask

        return score_dict

    def detect(self, tokenized_text: torch.Tensor = None,
               return_prediction: bool = True,
               return_scores: bool = True,
               z_threshold: Optional[float] = None,
               **kwargs) -> Dict:
        """
        Detect watermark in tokenized text.

        Args:
            tokenized_text: Token IDs
            return_prediction: Whether to return binary prediction
            return_scores: Whether to return scores
            z_threshold: Override z-score threshold

        Returns:
            Dictionary with detection results
        """
        if return_prediction:
            kwargs["return_p_value"] = True

        output_dict = {}
        score_dict = self._score_sequence(tokenized_text, **kwargs)

        if return_scores:
            output_dict.update(score_dict)

        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need threshold for prediction"

            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict
