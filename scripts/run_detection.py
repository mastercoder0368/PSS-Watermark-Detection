"""
Script to run watermark detection on text files.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.detector import WatermarkDetector
from src.detection.file_splitter import FileSplitter


def main():
    parser = argparse.ArgumentParser(description='Run Watermark Detection')
    parser.add_argument('--input-dir', required=True, help='Directory with P-files')
    parser.add_argument('--output-dir', required=True, help='Output directory for D-files')
    parser.add_argument('--text-type', choices=['ai', 'human'], required=True,
                        help='Type of text being processed')
    parser.add_argument('--split-first', action='store_true',
                        help='Split paraphrased CSV into P-files first')
    parser.add_argument('--exp-config', default='configs/experiment_config.yaml',
                        help='Path to experiment configuration')
    parser.add_argument('--model-config', default='configs/model_config.yaml',
                        help='Path to model configuration')

    args = parser.parse_args()

    # Load configurations
    with open(args.exp_config, 'r') as f:
        exp_config = yaml.safe_load(f)
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Split files if requested
    if args.split_first:
        print("Splitting paraphrased CSV into P-files...")
        splitter = FileSplitter()
        splitter.split_paraphrased_csv(
            input_dir=args.input_dir,
            text_type=args.text_type
        )

    # Initialize detector
    detector = WatermarkDetector(
        model_config=model_config,
        watermark_params=model_config['watermark_params']
    )

    # Process all P-files
    print(f"Running detection on {args.text_type} texts...")
    detector.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        text_type=args.text_type
    )

    print(f"\n✅ Detection complete!")
    print(f"  D-files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()