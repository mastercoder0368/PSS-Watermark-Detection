"""
Watermark Processor
Generates watermarked text using language models.
"""

import os
import torch
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm

from .watermark_base import WatermarkBase


class WatermarkLogitsProcessor(WatermarkBase):
    """Logits processor for applying watermarks during generation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.Tensor,
                            greenlist_token_ids: list) -> torch.Tensor:
        """
        Create mask for green tokens.

        Args:
            scores: Logit scores
            greenlist_token_ids: List of green token IDs per batch

        Returns:
            Boolean mask for green tokens
        """
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        return green_tokens_mask.bool()

    def _bias_greenlist_logits(self, scores: torch.Tensor,
                              greenlist_mask: torch.Tensor,
                              greenlist_bias: float) -> torch.Tensor:
        """
        Apply bias to green token logits.

        Args:
            scores: Logit scores
            greenlist_mask: Mask for green tokens
            greenlist_bias: Bias value (delta)

        Returns:
            Biased scores
        """
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Process logits during generation.

        Args:
            input_ids: Input token IDs
            scores: Logit scores

        Returns:
            Modified scores
        """
        # Initialize RNG if needed
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        # Process each sample in batch
        batched_greenlist_ids = []
        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids.append(greenlist_ids)

        # Apply watermark
        green_tokens_mask = self._calc_greenlist_mask(scores, batched_greenlist_ids)
        scores = self._bias_greenlist_logits(scores, green_tokens_mask, self.delta)

        return scores


class WatermarkProcessor:
    """Main processor for generating watermarked texts."""

    def __init__(self, model_config: Dict = None,
                 watermark_params: Dict = None,
                 generation_params: Dict = None):
        """
        Initialize watermark processor.

        Args:
            model_config: Model configuration
            watermark_params: Watermark parameters
            generation_params: Generation parameters
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Default configurations
        if model_config is None:
            model_config = {
                'name': 'meta-llama/Llama-2-7b-hf',
                'torch_dtype': 'float16',
                'device_map': 'auto',
                'use_fast_tokenizer': True
            }

        if watermark_params is None:
            watermark_params = {
                'gamma': 0.25,
                'delta': 1.5,
                'z_threshold': 4.0,
                'hash_key': 15485863,
                'seeding_scheme': 'simple_1',
                'select_green_tokens': True
            }

        if generation_params is None:
            generation_params = {
                'max_new_tokens': 3000,
                'min_new_tokens': 2800,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'repetition_penalty': 1.2,
                'no_repeat_ngram_size': 3,
                'early_stopping': True
            }

        self.model_config = model_config
        self.watermark_params = watermark_params
        self.generation_params = generation_params

        self.model = None
        self.tokenizer = None
        self.watermark_processor = None

    def setup_model(self):
        """Load model and tokenizer."""
        print(f"Loading model {self.model_config['name']}...")

        # Handle HuggingFace token
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            login(token=hf_token)
            print("✓ Logged in to HuggingFace")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['name'],
            use_fast=self.model_config.get('use_fast_tokenizer', True),
            token=hf_token
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        torch_dtype = getattr(torch, self.model_config.get('torch_dtype', 'float16'))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['name'],
            torch_dtype=torch_dtype,
            device_map=self.model_config.get('device_map', 'auto'),
            token=hf_token
        )

        # Create watermark processor
        self.watermark_processor = WatermarkLogitsProcessor(
            vocab=list(range(len(self.tokenizer))),
            gamma=self.watermark_params['gamma'],
            delta=self.watermark_params['delta'],
            seeding_scheme=self.watermark_params['seeding_scheme'],
            hash_key=self.watermark_params['hash_key'],
            select_green_tokens=self.watermark_params['select_green_tokens']
        )

        print(f"✓ Model loaded on {self.device}")

    def generate_watermarked_text(self, prompt: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate watermarked text from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (generated_text, only_generated_ids, full_ids)
        """
        if self.model is None:
            self.setup_model()

        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # Limit input length
        max_input_tokens = 2000
        if input_ids.shape[1] > max_input_tokens:
            input_ids = input_ids[:, :max_input_tokens]
            print(f"Input truncated to {max_input_tokens} tokens")

        # Set up watermark processor RNG
        self.watermark_processor.rng = torch.Generator(device=self.device)

        # Generate with watermark
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.generation_params['max_new_tokens'],
            min_new_tokens=self.generation_params['min_new_tokens'],
            do_sample=self.generation_params['do_sample'],
            temperature=self.generation_params['temperature'],
            top_p=self.generation_params['top_p'],
            repetition_penalty=self.generation_params['repetition_penalty'],
            no_repeat_ngram_size=self.generation_params['no_repeat_ngram_size'],
            logits_processor=[self.watermark_processor],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=self.generation_params['early_stopping']
        )

        # Extract generated text
        prompt_length = input_ids.shape[1]
        only_generated_ids = generated_ids[0][prompt_length:]
        generated_text = self.tokenizer.decode(only_generated_ids, skip_special_tokens=True)

        return generated_text, only_generated_ids, generated_ids[0]

    def process_csv(self, input_csv_path: str, output_text_csv_path: str,
                   output_bits_csv_path: str, max_rows: Optional[int] = None,
                   force_restart: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process CSV file and apply watermarks.

        Args:
            input_csv_path: Input CSV path
            output_text_csv_path: Output path for watermarked texts
            output_bits_csv_path: Output path for watermark bits
            max_rows: Maximum rows to process
            force_restart: Whether to restart from beginning

        Returns:
            Tuple of (text_df, bits_df)
        """
        print(f"Processing: {input_csv_path}")

        # Load input
        df = pd.read_csv(input_csv_path)
        if max_rows:
            df = df.head(max_rows)

        # Check for resume
        start_idx = 0
        write_mode = 'w'
        header = True

        if not force_restart and os.path.exists(output_text_csv_path):
            try:
                existing_text_df = pd.read_csv(output_text_csv_path)
                existing_bits_df = pd.read_csv(output_bits_csv_path)

                if len(existing_text_df) == len(existing_bits_df):
                    start_idx = len(existing_text_df)
                    write_mode = 'a'
                    header = False
                    print(f"Resuming from row {start_idx}")
            except:
                pass

        if start_idx >= len(df):
            print("✓ Already fully processed")
            return pd.read_csv(output_text_csv_path), pd.read_csv(output_bits_csv_path)

        # Setup model if needed
        if self.model is None:
            self.setup_model()

        # Import detector for validation
        from .watermark_detector import WatermarkDetector
        detector = WatermarkDetector(
            vocab=list(range(len(self.tokenizer))),
            gamma=self.watermark_params['gamma'],
            delta=self.watermark_params['delta'],
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=self.watermark_params['z_threshold'],
            seeding_scheme=self.watermark_params['seeding_scheme']
        )

        # Process texts
        for idx in tqdm(range(start_idx, len(df)), desc="Watermarking"):
            input_text = df.iloc[idx]["text"]

            # Generate watermarked text
            generated_text, only_generated_ids, full_ids = self.generate_watermarked_text(input_text)

            # Detect watermark for validation
            detection_results = detector.detect(
                tokenized_text=full_ids,
                return_prediction=True,
                return_scores=True,
                return_green_token_mask=True
            )

            # Convert mask to bit string
            bit_list = [1 if bit else 0 for bit in detection_results['green_token_mask']]
            bit_string = ','.join(map(str, bit_list))

            # Calculate word counts
            original_word_count = len(input_text.split())
            generated_word_count = len(generated_text.split())

            # Create dataframes for this row
            text_row_df = pd.DataFrame({
                'original_text': [input_text],
                'generated_text': [generated_text],
                'original_word_count': [original_word_count],
                'generated_word_count': [generated_word_count]
            })

            bits_row_df = pd.DataFrame({
                'watermark_bits': [bit_string],
                'z_score': [detection_results['z_score']]
            })

            # Save incrementally
            text_row_df.to_csv(output_text_csv_path, mode=write_mode,
                              header=header, index=False)
            bits_row_df.to_csv(output_bits_csv_path, mode=write_mode,
                              header=header, index=False)

            write_mode = 'a'
            header = False

        # Return final dataframes
        return pd.read_csv(output_text_csv_path), pd.read_csv(output_bits_csv_path)
