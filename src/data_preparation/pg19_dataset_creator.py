"""
PG-19 Dataset Creator
Creates dataset from PG-19 books for watermarking experiments.
"""

import os
import random
import pandas as pd
import re
import zipfile
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm


class PG19DatasetCreator:
    """Creates datasets from PG-19 books."""

    def __init__(self, output_dir: str = "data", config: dict = None):
        """
        Initialize dataset creator.

        Args:
            output_dir: Directory to save datasets
            config: Configuration dictionary from experiment_config.yaml
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Default configuration
        self.config = {
            'num_samples': 1000,
            'skip_lines': 200,
            'words_per_sample': 4000,
            'split_ratio': 0.5
        }

        # Update with provided config
        if config:
            self.config.update(config)

        self.random_seed = 42
        random.seed(self.random_seed)

    def setup_kaggle_credentials(self, kaggle_json_path: str = "kaggle.json") -> bool:
        """
        Set up Kaggle API credentials.

        Args:
            kaggle_json_path: Path to kaggle.json file

        Returns:
            True if successful, False otherwise
        """
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)

        if Path(kaggle_json_path).exists():
            import shutil
            shutil.copy(kaggle_json_path, kaggle_dir / "kaggle.json")
            os.chmod(kaggle_dir / "kaggle.json", 0o600)
            print("✓ Kaggle API credentials configured")
            return True
        else:
            print(f"Warning: {kaggle_json_path} not found")
            return False

    def download_and_extract(self):
        """Download and extract PG-19 dataset from Kaggle."""
        print("Downloading PG-19 dataset from Kaggle...")
        os.system(
            f"kaggle datasets download -d tunguz/the-pg19-language-modeling-benchmark-dataset -p {self.output_dir}")

        zip_path = self.output_dir / "the-pg19-language-modeling-benchmark-dataset.zip"
        if not zip_path.exists():
            raise FileNotFoundError("Failed to download dataset")

        print("Extracting train folder...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            train_files = [f for f in zip_ref.namelist()
                           if f.startswith('train/') and f.endswith('.txt')]
            print(f"Found {len(train_files)} .txt files in train folder")

            for file in tqdm(train_files, desc="Extracting files"):
                zip_ref.extract(file, self.output_dir)

        # Remove zip file
        zip_path.unlink()
        print("✅ Extraction completed")

        return list((self.output_dir / "train" / "train").glob("*.txt"))

    def extract_words_from_file(self, file_path: Path) -> Tuple[List[str], bool]:
        """
        Extract words from a file after skipping specified lines.

        Args:
            file_path: Path to text file

        Returns:
            Tuple of (extracted_words_list, success_flag)
        """
        skip_lines = self.config['skip_lines']
        words_to_extract = self.config['words_per_sample']

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Skip initial lines
                for _ in range(skip_lines):
                    line = f.readline()
                    if not line:  # End of file reached
                        return [], False

                # Collect words
                words = []
                for line in f:
                    line = re.sub(r'\s+', ' ', line.strip())
                    if line:
                        line_words = line.split()
                        words.extend(line_words)

                        if len(words) >= words_to_extract:
                            return words[:words_to_extract], True

                return words, False

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return [], False

    def check_file_viability(self, file_path: Path) -> bool:
        """
        Quick check if file has enough content.

        Args:
            file_path: Path to text file

        Returns:
            True if file has enough content, False otherwise
        """
        skip_lines = self.config['skip_lines']
        min_words = self.config['words_per_sample']

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Skip initial lines
                for _ in range(skip_lines):
                    if not f.readline():
                        return False

                # Count words in remaining content
                word_count = 0
                for line in f:
                    word_count += len(line.split())
                    if word_count >= min_words:
                        return True

                return False
        except:
            return False

    def create_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create dataset from PG-19 files.

        Returns:
            Tuple of (human_df, ai_df)
        """
        train_dir = self.output_dir / "train" / "train"

        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

        all_txt_files = list(train_dir.glob("*.txt"))
        print(f"Found {len(all_txt_files)} text files")

        # Find viable files
        print("Checking file viability...")
        viable_files = []
        for file_path in tqdm(all_txt_files, desc="Checking files"):
            if self.check_file_viability(file_path):
                viable_files.append(file_path)
                if len(viable_files) >= self.config['num_samples']:
                    break

        if len(viable_files) < self.config['num_samples']:
            print(f"Warning: Only {len(viable_files)} viable files found")

        # Process files
        human_data = []
        ai_data = []

        files_to_process = viable_files[:self.config['num_samples']]
        print(f"\nProcessing {len(files_to_process)} files...")

        for file_path in tqdm(files_to_process, desc="Processing"):
            words, success = self.extract_words_from_file(file_path)

            if success and len(words) == self.config['words_per_sample']:
                # Split words
                midpoint = self.config['words_per_sample'] // 2
                human_words = words[:midpoint]
                ai_words = words[midpoint:]

                # Create text
                human_text = ' '.join(human_words)
                ai_text = ' '.join(ai_words)

                # Add to data
                human_data.append({
                    'text': human_text,
                    'filename': file_path.name,
                    'word_count': len(human_words)
                })

                ai_data.append({
                    'text': ai_text,
                    'filename': file_path.name,
                    'word_count': len(ai_words)
                })

        print(f"✓ Successfully processed: {len(human_data)} files")

        # Create DataFrames
        human_df = pd.DataFrame(human_data)
        ai_df = pd.DataFrame(ai_data)

        # Save to CSV
        human_df.to_csv(self.output_dir / "human_text.csv", index=False)
        ai_df.to_csv(self.output_dir / "ai_input_text.csv", index=False)

        print(f"✓ Saved human_text.csv ({len(human_df)} samples)")
        print(f"✓ Saved ai_input_text.csv ({len(ai_df)} samples)")

        return human_df, ai_df

    def run(self):
        """Run the complete dataset creation pipeline."""
        # Check if train directory exists
        train_dir = self.output_dir / "train" / "train"

        if not train_dir.exists():
            print("Train directory not found. Please run with --download flag or download manually.")
            return None, None

        return self.create_dataset()

