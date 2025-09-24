"""
Watermark Detector
Detects watermarks in texts and generates detection metrics.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to import watermarking modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from watermarking.watermark_base import WatermarkBase
from watermarking.watermark_detector import WatermarkDetector as BaseDetector


class WatermarkDetector:
    """Detects watermarks in texts and saves detection results."""

    def __init__(self, model_config: dict = None, watermark_params: dict = None):
        """
        Initialize watermark detector.

        Args:
            model_config: Model configuration dictionary
            watermark_params: Watermark parameters dictionary
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Default configurations
        if model_config is None:
            model_config = {
                'name': 'meta-llama/Llama-2-7b-hf',
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

        self.model_config = model_config
        self.watermark_params = watermark_params

        # Initialize tokenizer and detector
        self._setup_detector()

    def _setup_detector(self):
        """Set up tokenizer and detector."""
        # Get HuggingFace token from environment
        hf_token = os.environ.get('HF_TOKEN', None)

        print(f"Loading tokenizer for {self.model_config['name']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['name'],
            use_fast=self.model_config.get('use_fast_tokenizer', True),
            token=hf_token
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create detector
        self.detector = BaseDetector(
            vocab=list(range(len(self.tokenizer))),
            gamma=self.watermark_params['gamma'],
            delta=self.watermark_params['delta'],
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=self.watermark_params['z_threshold'],
            seeding_scheme=self.watermark_params['seeding_scheme']
        )

        print(f"✓ Detector initialized (vocab size: {len(self.tokenizer)})")
        print(f"  Parameters: gamma={self.watermark_params['gamma']}, "
              f"delta={self.watermark_params['delta']}, "
              f"z_threshold={self.watermark_params['z_threshold']}")

    def detect_watermark(self, text: str) -> Dict:
        """
        Detect watermark in a single text.

        Args:
            text: Input text

        Returns:
            Dictionary with detection results
        """
        try:
            # Tokenize
            input_ids = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"][0].to(self.device)

            # Check minimum length
            if len(input_ids) <= self.detector.min_prefix_len:
                return {
                    'watermark_bits': '',
                    'z_score': 0.0,
                    'green_fraction': 0.0,
                    'green_token_count': 0,
                    'num_tokens_scored': 0
                }

            # Detect watermark
            detection_results = self.detector.detect(
                tokenized_text=input_ids,
                return_prediction=True,
                return_scores=True,
                return_green_token_mask=True
            )

            # Convert mask to bit string
            if 'green_token_mask' in detection_results:
                bit_list = [1 if bit else 0 for bit in detection_results['green_token_mask']]
                bit_string = ','.join(map(str, bit_list))
            else:
                bit_string = ''

            return {
                'watermark_bits': bit_string,
                'z_score': detection_results.get('z_score', 0.0),
                'green_fraction': detection_results.get('green_fraction', 0.0),
                'green_token_count': detection_results.get('num_green_tokens', 0),
                'num_tokens_scored': detection_results.get('num_tokens_scored', 0)
            }

        except Exception as e:
            print(f"Error in detection: {e}")
            return {
                'watermark_bits': 'ERROR',
                'z_score': 0.0,
                'green_fraction': 0.0,
                'green_token_count': 0,
                'num_tokens_scored': 0
            }

    def process_file(self, input_file: str, output_file: str,
                     resume: bool = True) -> pd.DataFrame:
        """
        Process a CSV file and detect watermarks.

        Args:
            input_file: Path to input CSV
            output_file: Path to output CSV
            resume: Whether to resume from existing progress

        Returns:
            DataFrame with detection results
        """
        print(f"\nProcessing: {input_file}")

        # Read input
        df = pd.read_csv(input_file)

        # Find text column
        text_column = None
        possible_columns = ['text', 'original_text', 'paraphrase_iter_1',
                            'paraphrase_iter_2', 'paraphrase_iter_3',
                            'paraphrase_iter_4', 'paraphrase_iter_5',
                            'paraphrase_iter_6', 'paraphrase_iter_7',
                            'paraphrase_iter_8']

        for col in df.columns:
            if col in possible_columns or 'text' in col.lower():
                text_column = col
                break

        if text_column is None:
            print(f"Using first column: {df.columns[0]}")
            text_column = df.columns[0]
        else:
            print(f"Using column: {text_column}")

        # Check for existing output
        start_idx = 0
        if resume and os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                start_idx = len(existing_df)
                print(f"Resuming from row {start_idx}")

                if start_idx >= len(df):
                    print("✓ File already fully processed")
                    return existing_df

                # Initialize with existing results
                results = {
                    'watermark_bits': existing_df['watermark_bits'].tolist(),
                    'z_score': existing_df['z_score'].tolist(),
                    'green_fraction': existing_df['green_fraction'].tolist(),
                    'green_token_count': existing_df['green_token_count'].tolist()
                }
            except:
                start_idx = 0
                results = {
                    'watermark_bits': [],
                    'z_score': [],
                    'green_fraction': [],
                    'green_token_count': []
                }
        else:
            results = {
                'watermark_bits': [],
                'z_score': [],
                'green_fraction': [],
                'green_token_count': []
            }

        # Process each text
        for idx in tqdm(range(start_idx, len(df)), desc="Detecting"):
            text = str(df.iloc[idx][text_column])
            detection = self.detect_watermark(text)

            results['watermark_bits'].append(detection['watermark_bits'])
            results['z_score'].append(detection['z_score'])
            results['green_fraction'].append(detection['green_fraction'])
            results['green_token_count'].append(detection['green_token_count'])

            # Save progress periodically
            if (idx + 1 - start_idx) % 50 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(output_file, index=False)

        # Save final results
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_file, index=False)

        print(f"✓ Saved results to: {output_file}")
        print(f"  Average z-score: {final_df['z_score'].mean():.2f}")
        print(f"  Average green fraction: {final_df['green_fraction'].mean():.3f}")

        return final_df

    def process_directory(self, input_dir: str, output_dir: str,
                          text_type: str = 'ai') -> None:
        """
        Process all P-files in a directory.

        Args:
            input_dir: Directory with P-files
            output_dir: Output directory for D-files
            text_type: 'ai' or 'human'
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Define file mappings
        if text_type == 'ai':
            file_mappings = [
                ('AI_Original_Text.csv', 'AI_original_zscore.csv'),
                ('AI_P1.csv', 'AI_D1.csv'),
                ('AI_P2.csv', 'AI_D2.csv'),
                ('AI_P3.csv', 'AI_D3.csv'),
                ('AI_P4.csv', 'AI_D4.csv'),
                ('AI_P5.csv', 'AI_D5.csv'),
                ('AI_P6.csv', 'AI_D6.csv'),
                ('AI_P7.csv', 'AI_D7.csv'),
                ('AI_P8.csv', 'AI_D8.csv')
            ]
        else:  # human
            file_mappings = [
                ('Human_Original_Text.csv', 'Human_original_zscore.csv'),
                ('Human_P1.csv', 'Human_D1.csv'),
                ('Human_P2.csv', 'Human_D2.csv'),
                ('Human_P3.csv', 'Human_D3.csv'),
                ('Human_P4.csv', 'Human_D4.csv'),
                ('Human_P5.csv', 'Human_D5.csv'),
                ('Human_P6.csv', 'Human_D6.csv'),
                ('Human_P7.csv', 'Human_D7.csv'),
                ('Human_P8.csv', 'Human_D8.csv')
            ]

        # Process each file
        for input_name, output_name in file_mappings:
            input_file = input_path / input_name
            output_file = output_path / output_name

            if input_file.exists():
                print(f"\nProcessing: {input_name} → {output_name}")
                self.process_file(str(input_file), str(output_file))
            else:
                print(f"Warning: {input_file} not found, skipping...")

        print(f"\n✓ All {text_type} files processed!")
