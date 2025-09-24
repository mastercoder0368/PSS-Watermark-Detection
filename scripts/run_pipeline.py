"""
Main pipeline script for PSS Watermark Detection experiments.
Runs the complete pipeline from dataset creation to PSS analysis.
"""
import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preparation.pg19_dataset_creator import PG19DatasetCreator
from src.data_preparation.text_length_adjuster import TextLengthAdjuster
from src.watermarking.watermark_processor import WatermarkProcessor
from src.paraphrasing.batch_paraphraser import BatchParaphraser
from src.detection.detector import WatermarkDetector
from src.detection.file_splitter import FileSplitter
from src.pss_method.data_combiner import DataCombiner
from src.pss_method.rolling_window_analyzer import RollingWindowAnalyzer
from src.pss_method.pss_classifier import PSSClassifier


def load_configs(experiment_config_path, model_config_path):
    """Load configuration files."""
    with open(experiment_config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    return exp_config, model_config


def main():
    parser = argparse.ArgumentParser(description='Run PSS Watermark Detection Pipeline')
    parser.add_argument('--exp-config', default='configs/experiment_config.yaml',
                        help='Path to experiment configuration')
    parser.add_argument('--model-config', default='configs/model_config.yaml',
                        help='Path to model configuration')
    parser.add_argument('--steps', nargs='+',
                        choices=['all', 'dataset', 'watermark', 'paraphrase',
                                 'detect', 'combine', 'rolling', 'pss'],
                        default=['all'], help='Which steps to run')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--results-dir', default='results', help='Results directory')

    args = parser.parse_args()

    # Load configurations
    exp_config, model_config = load_configs(args.exp_config, args.model_config)

    # Create directories
    Path(args.data_dir).mkdir(exist_ok=True)
    Path(args.results_dir).mkdir(exist_ok=True)

    steps_to_run = args.steps if args.steps != ['all'] else [
        'dataset', 'watermark', 'paraphrase', 'detect', 'combine', 'rolling', 'pss'
    ]

    print("=" * 70)
    print("PSS WATERMARK DETECTION PIPELINE")
    print("=" * 70)
    print(f"Steps to run: {', '.join(steps_to_run)}")
    print("=" * 70)

    # Step 1: Dataset Creation
    if 'dataset' in steps_to_run:
        print("\n[Step 1] Creating PG-19 Dataset...")
        creator = PG19DatasetCreator(
            output_dir=args.data_dir,
            config=exp_config['dataset']
        )
        creator.run()
        print("✅ Dataset creation complete")

    # Step 2: Watermark Generation
    if 'watermark' in steps_to_run:
        print("\n[Step 2] Generating Watermarked Text...")
        processor = WatermarkProcessor(
            data_dir=args.data_dir,
            model_config=model_config,
            exp_config=exp_config
        )
        processor.run()
        print("✅ Watermark generation complete")

    # Step 3: Paraphrasing
    if 'paraphrase' in steps_to_run:
        print("\n[Step 3] Paraphrasing Texts...")
        paraphraser = BatchParaphraser(
            data_dir=args.data_dir,
            model_config=model_config,
            exp_config=exp_config
        )
        # Paraphrase both AI and Human texts
        paraphraser.paraphrase_ai_texts()
        paraphraser.paraphrase_human_texts()
        print("✅ Paraphrasing complete")

    # Step 4: Detection
    if 'detect' in steps_to_run:
        print("\n[Step 4] Running Watermark Detection...")
        detector = WatermarkDetector(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            model_config=model_config,
            exp_config=exp_config
        )
        detector.run()
        print("✅ Detection complete")

    # Step 5: Combine AI and Human results
    if 'combine' in steps_to_run:
        print("\n[Step 5] Combining AI and Human Results...")
        combiner = DataCombiner(
            results_dir=args.results_dir,
            exp_config=exp_config
        )
        combiner.run()
        print("✅ Data combination complete")

    # Step 6: Rolling Window Analysis
    if 'rolling' in steps_to_run:
        print("\n[Step 6] Performing Rolling Window Analysis...")
        analyzer = RollingWindowAnalyzer(
            results_dir=args.results_dir,
            exp_config=exp_config
        )
        analyzer.run()
        print("✅ Rolling window analysis complete")

    # Step 7: PSS Classification
    if 'pss' in steps_to_run:
        print("\n[Step 7] Running PSS Classification...")
        classifier = PSSClassifier(
            results_dir=args.results_dir,
            exp_config=exp_config
        )
        classifier.run()
        print("✅ PSS classification complete")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Results saved in:", args.results_dir)
    print("=" * 70)


if __name__ == '__main__':
    main()
