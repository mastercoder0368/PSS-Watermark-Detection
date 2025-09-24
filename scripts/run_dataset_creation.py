"""
Script to create PG-19 dataset for watermarking experiments.
"""

import os
import sys
import argparse
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preparation.pg19_dataset_creator import PG19DatasetCreator
from src.data_preparation.text_length_adjuster import TextLengthAdjuster


def main():
    parser = argparse.ArgumentParser(description='Create PG-19 Dataset')
    parser.add_argument('--config', default='configs/experiment_config.yaml',
                        help='Path to experiment configuration')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--num-samples', type=int, help='Override number of samples')
    parser.add_argument('--download', action='store_true', help='Download dataset from Kaggle')
    parser.add_argument('--kaggle-json', default='kaggle.json', help='Path to Kaggle credentials')
    parser.add_argument('--create-variations', action='store_true',
                        help='Create text length variations')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset_config = config['dataset']

    # Override with command line arguments
    if args.num_samples:
        dataset_config['num_samples'] = args.num_samples

    # Create dataset
    creator = PG19DatasetCreator(
        output_dir=args.output_dir,
        config=dataset_config
    )

    if args.download:
        creator.setup_kaggle_credentials(args.kaggle_json)
        creator.download_and_extract()

    # Create main dataset
    human_df, ai_df = creator.create_dataset()

    print(f"✅ Created {len(human_df)} human text samples")
    print(f"✅ Created {len(ai_df)} AI input text samples")

    # Create length variations if requested
    if args.create_variations:
        adjuster = TextLengthAdjuster()

        for length in config['text_lengths']:
            print(f"\nCreating {length}-word variations...")

            # Adjust human texts
            human_adjusted = adjuster.adjust_text_length(human_df, length)
            human_adjusted.to_csv(f"{args.output_dir}/human_text_{length}.csv", index=False)

            # Adjust AI texts
            ai_adjusted = adjuster.adjust_text_length(ai_df, length)
            ai_adjusted.to_csv(f"{args.output_dir}/ai_input_text_{length}.csv", index=False)

            print(f"✅ Saved {length}-word variations")

    print("\n✅ Dataset creation complete!")


if __name__ == '__main__':
    main()
