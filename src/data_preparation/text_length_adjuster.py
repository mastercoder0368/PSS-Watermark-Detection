"""
Text Length Adjuster
Adjusts text lengths for different experimental conditions.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class TextLengthAdjuster:
    """Adjusts text lengths by trimming words."""

    @staticmethod
    def adjust_text_length(df: pd.DataFrame, target_words: int,
                           text_column: str = 'text',
                           output_column: str = None) -> pd.DataFrame:
        """
        Adjust text length to target word count.

        Args:
            df: Input DataFrame with text
            target_words: Target number of words
            text_column: Name of column containing text
            output_column: Name for output column (if None, creates new columns)

        Returns:
            DataFrame with adjusted text
        """
        df_adjusted = df.copy()

        # Define output columns
        if output_column is None:
            text_out = 'text_cleaned'
            count_out = 'new_word_count'
        else:
            text_out = output_column
            count_out = f'{output_column}_word_count'

        def trim_text(text):
            """Trim text to target word count."""
            if pd.isna(text):
                return '', 0

            words = str(text).split()
            current_count = len(words)

            # If already at or below target, keep as is
            if current_count <= target_words:
                return text, current_count

            # Trim to target
            trimmed_words = words[:target_words]
            return ' '.join(trimmed_words), len(trimmed_words)

        # Apply trimming
        results = df_adjusted[text_column].apply(lambda x: pd.Series(trim_text(x)))
        df_adjusted[text_out] = results[0]
        df_adjusted[count_out] = results[1]

        return df_adjusted

    @staticmethod
    def process_watermarked_text(df: pd.DataFrame, target_words: int) -> pd.DataFrame:
        """
        Process watermarked text DataFrame with special handling.

        Args:
            df: DataFrame with 'generated_text' column
            target_words: Target number of words

        Returns:
            Adjusted DataFrame
        """
        return TextLengthAdjuster.adjust_text_length(
            df,
            target_words,
            text_column='generated_text',
            output_column='text_cleaned'
        )

    @staticmethod
    def process_human_text(df: pd.DataFrame, target_words: int) -> pd.DataFrame:
        """
        Process human text DataFrame.

        Args:
            df: DataFrame with 'text' column
            target_words: Target number of words

        Returns:
            Adjusted DataFrame
        """
        return TextLengthAdjuster.adjust_text_length(
            df,
            target_words,
            text_column='text',
            output_column='text_cleaned'
        )

    @staticmethod
    def create_length_variations(base_csv_path: str,
                                 variations=None,
                                 text_type: str = 'human') -> None:
        """
        Create multiple length variations from base CSV.

        Args:
            base_csv_path: Path to base CSV file
            variations: List of target word counts
            text_type: 'human' or 'ai' to determine processing method
        """
        if variations is None:
            variations = [1500, 1000, 500, 300]
        base_path = Path(base_csv_path)
        df = pd.read_csv(base_path)

        output_dir = base_path.parent
        base_name = base_path.stem.replace('_text', '')

        for length in variations:
            print(f"Creating {length}-word variation...")

            if text_type == 'ai' or 'watermarked' in base_name.lower():
                adjusted_df = TextLengthAdjuster.process_watermarked_text(df, length)
            else:
                adjusted_df = TextLengthAdjuster.process_human_text(df, length)

            # Save with appropriate naming
            if 'watermarked' in base_name.lower():
                output_name = f"ai_watermarked_{length}.csv"
            else:
                output_name = f"{base_name}_{length}.csv"

            output_path = output_dir / output_name
            adjusted_df.to_csv(output_path, index=False)
            print(f"✓ Saved: {output_path}")

    @staticmethod
    def batch_process_variations(data_dir: str, variations: list = [1500, 1000, 500, 300]):
        """
        Process all text files in directory to create variations.

        Args:
            data_dir: Directory containing CSV files
            variations: List of target word counts
        """
        data_path = Path(data_dir)

        # Process human text
        human_file = data_path / "human_text.csv"
        if human_file.exists():
            print("\nProcessing human text variations...")
            TextLengthAdjuster.create_length_variations(
                str(human_file), variations, 'human'
            )

        # Process AI watermarked text
        ai_file = data_path / "ai_watermarked_text.csv"
        if ai_file.exists():
            print("\nProcessing AI watermarked text variations...")
            TextLengthAdjuster.create_length_variations(
                str(ai_file), variations, 'ai'
            )
