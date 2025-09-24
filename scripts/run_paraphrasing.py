"""
Script to paraphrase texts for multiple iterations.
"""

import os
import sys
import argparse
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.paraphrasing.batch_paraphraser import BatchParaphraser


def main():
    parser = argparse.ArgumentParser(description='Paraphrase Texts')
    parser.add_argument('--input-csv', required=True, help='Input CSV file')
    parser.add_argument('--output-csv', required=True, help='Output CSV file')
    parser.add_argument('--text-type', choices=['ai', 'human'], required=True,
                        help='Type of text being processed')
    parser.add_argument('--model-path', help='Path to GGUF model file')
    parser.add_argument('--iterations', type=int, default=8,
                        help='Number of paraphrasing iterations')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for processing')
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

    # Initialize paraphraser
    paraphraser = BatchParaphraser(
        model_config=model_config['paraphrase_model'],
        paraphrase_params=model_config['paraphrase_params'],
        iterations=args.iterations or exp_config['paraphrasing']['iterations'],
        batch_size=args.batch_size or exp_config['paraphrasing']['batch_size']
    )

    # Set text column based on type
    if args.text_type == 'ai':
        text_column = 'generated_text'
        word_count_column = 'generated_word_count'
    else:  # human
        text_column = 'text_cleaned'
        word_count_column = 'new_word_count'

    # Process the file
    print(f"Paraphrasing {args.text_type} texts from: {args.input_csv}")
    paraphraser.process_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        text_column=text_column,
        word_count_column=word_count_column,
        model_path=args.model_path
    )

    print(f"\n✅ Paraphrasing complete!")
    print(f"  Output saved to: {args.output_csv}")


if __name__ == "__main__":
    main()