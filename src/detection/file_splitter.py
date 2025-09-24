"""
File Splitter
Splits paraphrased CSV files into individual P-files for detection.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict


class FileSplitter:
    """Splits paraphrased CSV files into individual iteration files."""

    @staticmethod
    def split_paraphrased_csv(input_dir: str, text_type: str = 'ai') -> Dict[str, Path]:
        """
        Split a paraphrased CSV into individual P-files.

        Args:
            input_dir: Directory containing paraphrased CSV
            text_type: 'ai' or 'human'

        Returns:
            Dictionary mapping P-file names to their paths
        """
        input_path = Path(input_dir)

        # Find the paraphrased file
        if text_type == 'ai':
            paraphrased_file = input_path / 'ai_paraphrased.csv'
            prefix = 'AI'
        else:
            paraphrased_file = input_path / 'human_paraphrased.csv'
            prefix = 'Human'

        if not paraphrased_file.exists():
            print(f"Warning: {paraphrased_file} not found")
            return {}

        print(f"Splitting {paraphrased_file}...")
        df = pd.read_csv(paraphrased_file)

        # Define column mappings
        column_mappings = {
            'original_text': f'{prefix}_Original_Text.csv',
            'paraphrase_iter_1': f'{prefix}_P1.csv',
            'paraphrase_iter_2': f'{prefix}_P2.csv',
            'paraphrase_iter_3': f'{prefix}_P3.csv',
            'paraphrase_iter_4': f'{prefix}_P4.csv',
            'paraphrase_iter_5': f'{prefix}_P5.csv',
            'paraphrase_iter_6': f'{prefix}_P6.csv',
            'paraphrase_iter_7': f'{prefix}_P7.csv',
            'paraphrase_iter_8': f'{prefix}_P8.csv'
        }

        output_files = {}

        # Create individual files
        for col, filename in column_mappings.items():
            if col in df.columns:
                output_path = input_path / filename
                df[[col]].to_csv(output_path, index=False)
                output_files[filename] = output_path
                print(f"✓ Created: {filename}")
            else:
                print(f"Warning: Column '{col}' not found in {paraphrased_file}")

        return output_files

    @staticmethod
    def combine_detection_results(detection_dir: str, output_file: str,
                                  text_type: str = 'ai') -> pd.DataFrame:
        """
        Combine multiple D-files into a single DataFrame.

        Args:
            detection_dir: Directory containing D-files
            output_file: Path for combined output
            text_type: 'ai' or 'human'

        Returns:
            Combined DataFrame
        """
        detection_path = Path(detection_dir)

        if text_type == 'ai':
            d_files = [
                'AI_original_zscore.csv',
                'AI_D1.csv', 'AI_D2.csv', 'AI_D3.csv', 'AI_D4.csv',
                'AI_D5.csv', 'AI_D6.csv', 'AI_D7.csv', 'AI_D8.csv'
            ]
        else:
            d_files = [
                'Human_original_zscore.csv',
                'Human_D1.csv', 'Human_D2.csv', 'Human_D3.csv', 'Human_D4.csv',
                'Human_D5.csv', 'Human_D6.csv', 'Human_D7.csv', 'Human_D8.csv'
            ]

        combined_data = []

        for idx, filename in enumerate(d_files):
            file_path = detection_path / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['depth'] = idx  # 0 for original, 1-8 for paraphrases
                df['text_type'] = text_type
                combined_data.append(df)
                print(f"✓ Added {filename}: {len(df)} rows")
            else:
                print(f"Warning: {file_path} not found")

        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            print(f"✓ Saved combined results to: {output_file}")
            return combined_df
        else:
            print("No data to combine")
            return pd.DataFrame()

