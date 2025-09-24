"""
Data Combiner
Combines AI and Human detection results for PSS analysis.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict


class DataCombiner:
    """Combines AI and Human detection results."""

    @staticmethod
    def combine_d_files(ai_dir: str, human_dir: str, output_dir: str) -> None:
        """
        Combine AI and Human D-files into combined files with labels.

        Args:
            ai_dir: Directory containing AI D-files
            human_dir: Directory containing Human D-files
            output_dir: Output directory for combined files
        """
        ai_path = Path(ai_dir)
        human_path = Path(human_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Define file pairs
        file_pairs = [
            ('AI_D1.csv', 'Human_D1.csv', 'combined_D1.csv'),
            ('AI_D2.csv', 'Human_D2.csv', 'combined_D2.csv'),
            ('AI_D3.csv', 'Human_D3.csv', 'combined_D3.csv'),
            ('AI_D4.csv', 'Human_D4.csv', 'combined_D4.csv'),
            ('AI_D5.csv', 'Human_D5.csv', 'combined_D5.csv'),
            ('AI_D6.csv', 'Human_D6.csv', 'combined_D6.csv'),
            ('AI_D7.csv', 'Human_D7.csv', 'combined_D7.csv'),
            ('AI_D8.csv', 'Human_D8.csv', 'combined_D8.csv'),
            ('AI_original_zscore.csv', 'Human_original_zscore.csv', 'combined_original_zscore.csv')
        ]

        # Columns to keep
        keep_cols = ['watermark_bits', 'z_score']

        # Process each pair
        for ai_file, human_file, output_file in file_pairs:
            ai_filepath = ai_path / ai_file
            human_filepath = human_path / human_file
            output_filepath = output_path / output_file

            if not ai_filepath.exists():
                print(f"Warning: {ai_filepath} not found, skipping...")
                continue
            if not human_filepath.exists():
                print(f"Warning: {human_filepath} not found, skipping...")
                continue

            # Load data
            df_ai = pd.read_csv(ai_filepath, usecols=keep_cols).copy()
            df_human = pd.read_csv(human_filepath, usecols=keep_cols).copy()

            # Add labels (1 for AI, 0 for Human)
            df_ai['label'] = 1
            df_human['label'] = 0

            # Combine
            df_combined = pd.concat([df_ai, df_human], ignore_index=True)

            # Add sequential ID
            df_combined.insert(0, 'id', range(1, len(df_combined) + 1))

            # Rename columns
            df_combined = df_combined.rename(columns={'watermark_bits': '0_1_sequence'})

            # Reorder columns
            df_combined = df_combined[['id', '0_1_sequence', 'z_score', 'label']]

            # Save
            df_combined.to_csv(output_filepath, index=False)
            print(f"✅ Created {output_file} with {len(df_combined)} rows "
                  f"({len(df_ai)} AI + {len(df_human)} Human)")

        print("\n✓ All files combined successfully!")

    @staticmethod
    def validate_combined_files(combined_dir: str) -> Dict[str, int]:
        """
        Validate that all combined files have consistent row counts.

        Args:
            combined_dir: Directory with combined files

        Returns:
            Dictionary with file names and row counts
        """
        combined_path = Path(combined_dir)

        expected_files = [
            'combined_D1.csv', 'combined_D2.csv', 'combined_D3.csv',
            'combined_D4.csv', 'combined_D5.csv', 'combined_D6.csv',
            'combined_D7.csv', 'combined_D8.csv', 'combined_original_zscore.csv'
        ]

        file_info = {}

        for filename in expected_files:
            filepath = combined_path / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                file_info[filename] = len(df)

                # Check label distribution
                ai_count = (df['label'] == 1).sum()
                human_count = (df['label'] == 0).sum()
                print(f"{filename}: {len(df)} rows ({ai_count} AI, {human_count} Human)")
            else:
                print(f"Warning: {filename} not found")
                file_info[filename] = 0

        # Check consistency
        row_counts = [count for count in file_info.values() if count > 0]
        if len(set(row_counts)) == 1:
            print(f"\n✅ All files have consistent row count: {row_counts[0]}")
        else:
            print(f"\n⚠️ Warning: Inconsistent row counts: {set(row_counts)}")

        return file_info

