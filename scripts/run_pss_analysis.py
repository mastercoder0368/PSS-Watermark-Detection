"""
Script to run complete PSS analysis pipeline.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pss_method.data_combiner import DataCombiner
from src.pss_method.rolling_window_analyzer import RollingWindowAnalyzer
from src.pss_method.pss_classifier import PSSClassifier


def main():
    parser = argparse.ArgumentParser(description='Run PSS Analysis')
    parser.add_argument('--ai-dir', required=True, help='Directory with AI D-files')
    parser.add_argument('--human-dir', required=True, help='Directory with Human D-files')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--window-size', type=int, default=50, help='Rolling window size')
    parser.add_argument('--stride', type=int, default=10, help='Rolling window stride')
    parser.add_argument('--exp-config', default='configs/experiment_config.yaml',
                        help='Path to experiment configuration')
    parser.add_argument('--skip-combine', action='store_true',
                        help='Skip combination step if already done')
    parser.add_argument('--skip-rolling', action='store_true',
                        help='Skip rolling window analysis if already done')

    args = parser.parse_args()

    # Load configuration
    with open(args.exp_config, 'r') as f:
        exp_config = yaml.safe_load(f)

    # Create output directories
    combined_dir = Path(args.output_dir) / 'combined'
    rolling_dir = Path(args.output_dir) / 'rolling_window_stats'
    pss_dir = Path(args.output_dir) / 'pss_results'

    for dir_path in [combined_dir, rolling_dir, pss_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Combine AI and Human D-files
    if not args.skip_combine:
        print("Step 1: Combining AI and Human D-files...")
        combiner = DataCombiner()
        combiner.combine_d_files(
            ai_dir=args.ai_dir,
            human_dir=args.human_dir,
            output_dir=str(combined_dir)
        )
        print("✓ Combination complete")

    # Step 2: Rolling Window Analysis
    if not args.skip_rolling:
        print("\nStep 2: Performing Rolling Window Analysis...")
        analyzer = RollingWindowAnalyzer(
            window_size=args.window_size or exp_config['rolling_window']['window_size'],
            stride=args.stride or exp_config['rolling_window']['stride'],
            gamma=exp_config['watermark_params']['gamma'] if 'watermark_params' in exp_config
            else 0.25
        )
        analyzer.process_directory(
            input_dir=str(combined_dir),
            output_dir=str(rolling_dir)
        )
        print("✓ Rolling window analysis complete")

    # Step 3: PSS Classification
    print("\nStep 3: Running PSS Classification...")
    classifier = PSSClassifier(
        rolling_dir=str(rolling_dir),
        output_dir=str(pss_dir),
        exp_config=exp_config
    )

    # Run experiments
    results = classifier.run_all_experiments()

    # Save results
    output_path = pss_dir / 'pss_results.csv'
    results.to_csv(output_path, index=False)

    print(f"\n✓ PSS analysis complete!")
    print(f"  Results saved to: {output_path}")

    # Print summary
    print("\nResults Summary:")
    print(results[['Experiment', 'Accuracy (%)', 'AUC', 'F1']])


if __name__ == "__main__":
    main()