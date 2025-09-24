"""
Script to generate watermarked texts from AI input texts.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.watermarking.watermark_processor import WatermarkProcessor


def main():
    parser = argparse.ArgumentParser(description='Generate Watermarked Texts')
    parser.add_argument('--input-csv', required=True, help='Input CSV with AI texts')
    parser.add_argument('--output-text-csv', required=True, help='Output CSV for watermarked texts')
    parser.add_argument('--output-bits-csv', required=True, help='Output CSV for watermark bits')
    parser.add_argument('--exp-config', default='configs/experiment_config.yaml',
                        help='Path to experiment configuration')
    parser.add_argument('--model-config', default='configs/model_config.yaml',
                        help='Path to model configuration')
    parser.add_argument('--max-rows', type=int, help='Process only first N rows')
    parser.add_argument('--force-restart', action='store_true',
                        help='Start from beginning, ignore existing progress')

    args = parser.parse_args()

    # Load configurations
    with open(args.exp_config, 'r') as f:
        exp_config = yaml.safe_load(f)
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    # Initialize processor
    processor = WatermarkProcessor(
        model_config=model_config['watermark_model'],
        watermark_params=model_config['watermark_params'],
        generation_params=model_config['generation_params']
    )

    # Process the CSV
    print(f"Processing: {args.input_csv}")
    text_df, bits_df = processor.process_csv(
        input_csv_path=args.input_csv,
        output_text_csv_path=args.output_text_csv,
        output_bits_csv_path=args.output_bits_csv,
        max_rows=args.max_rows,
        force_restart=args.force_restart
    )

    print(f"\n✅ Watermarking complete!")
    print(f"  Text output: {args.output_text_csv}")
    print(f"  Bits output: {args.output_bits_csv}")
    print(f"  Total processed: {len(text_df)} texts")


if __name__ == "__main__":
    main()