"""
Rolling Window Analyzer
Performs rolling window analysis on 0-1 sequences.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis
from typing import List, Tuple, Dict


class RollingWindowAnalyzer:
    """Performs rolling window analysis on detection results."""

    def __init__(self, window_size: int = 50, stride: int = 10, gamma: float = 0.25):
        """
        Initialize rolling window analyzer.

        Args:
            window_size: Size of rolling window
            stride: Stride for rolling window
            gamma: Watermark gamma parameter
        """
        self.window_size = window_size
        self.stride = stride
        self.gamma = gamma

    def sliding_window(self, seq: str) -> List[Tuple[int, int]]:
        """
        Generate sliding windows with edge expansion for runs of 1s.

        Args:
            seq: Binary sequence string

        Returns:
            List of (start, end) tuples for windows
        """
        n = len(seq)
        windows = []

        for start_pos in range(0, n, self.stride):
            end_pos = min(start_pos + self.window_size, n)

            # Expand backward if window starts inside a run of 1s
            start = start_pos
            while start > 0 and seq[start] == "1":
                start -= 1

            # Expand forward if window ends inside a run of 1s
            end = end_pos
            while end < n and seq[end - 1] == "1":
                end += 1

            windows.append((start, end))

            if end_pos == n:
                break  # Last window

        return windows

    def compute_z_score(self, window_bits: List[str]) -> float:
        """
        Compute watermark z-score for a window.

        Args:
            window_bits: List of '0'/'1' characters

        Returns:
            Z-score
        """
        T = len(window_bits)
        if T == 0:
            return 0.0

        observed = window_bits.count("1")
        expected = self.gamma * T
        std_dev = np.sqrt(T * self.gamma * (1 - self.gamma))

        if std_dev == 0:
            return 0.0

        return (observed - expected) / std_dev

    def longest_run_and_freq(self, window_bits: List[str]) -> Tuple[int, int]:
        """
        Find longest run of 1s and its frequency.

        Args:
            window_bits: List of '0'/'1' characters

        Returns:
            Tuple of (longest_run_length, frequency)
        """
        # Join bits and split by 0s to find runs of 1s
        bit_string = "".join(window_bits)
        runs = [len(r) for r in bit_string.split("0") if r]

        if not runs:
            return 0, 0

        longest = max(runs)
        freq = runs.count(longest)

        return longest, freq

    def compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Compute summary statistics for a list of values.

        Args:
            values: List of numeric values

        Returns:
            Dictionary with statistics
        """
        arr = np.array(values, dtype=float)

        if arr.size == 0:
            return {
                'mean': 0.0, 'var': 0.0, 'min': 0.0,
                'max': 0.0, 'skew': 0.0, 'kurtosis': 0.0
            }

        return {
            'mean': float(arr.mean()),
            'var': float(arr.var(ddof=0)),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'skew': float(skew(arr)) if arr.size > 2 else 0.0,
            'kurtosis': float(kurtosis(arr)) if arr.size > 3 else 0.0
        }

    def autocorrelation(self, z_scores: List[float], lag: int) -> float:
        """
        Compute autocorrelation at specified lag.

        Args:
            z_scores: List of z-scores
            lag: Lag value

        Returns:
            Autocorrelation coefficient
        """
        z = np.array(z_scores, dtype=float)

        if z.size <= lag:
            return 0.0

        z_centered = z - z.mean()
        denominator = (z_centered ** 2).sum()

        if denominator == 0:
            return 0.0

        numerator = (z_centered[lag:] * z_centered[:-lag]).sum()
        return float(numerator / denominator)

    def process_sequence(self, sequence: str, global_z: float,
                         label: int, row_id: int) -> Dict:
        """
        Process a single 0-1 sequence with rolling windows.

        Args:
            sequence: Binary sequence string
            global_z: Global z-score
            label: Label (0 or 1)
            row_id: Row identifier

        Returns:
            Dictionary with all features
        """
        # Lists to store window metrics
        z_scores = []
        longest_runs = []
        frequencies = []

        # Process each window
        for start, end in self.sliding_window(sequence):
            window = list(sequence[start:end])

            # Compute metrics
            z = self.compute_z_score(window)
            run, freq = self.longest_run_and_freq(window)

            z_scores.append(z)
            longest_runs.append(run)
            frequencies.append(freq)

        # Compute summary statistics
        z_stats = self.compute_statistics(z_scores)
        run_stats = self.compute_statistics(longest_runs)
        freq_stats = self.compute_statistics(frequencies)

        # Compute autocorrelation
        z_autocorr_lag1 = self.autocorrelation(z_scores, 1)
        z_autocorr_lag2 = self.autocorrelation(z_scores, 2)

        # Build output dictionary
        result = {
            'id': row_id,
            '0_1_sequence': sequence,
            'z_score': global_z,
            'label': label,

            # Z-score statistics
            'z_mean': z_stats['mean'],
            'z_var': z_stats['var'],
            'z_min': z_stats['min'],
            'z_max': z_stats['max'],
            'z_skew': z_stats['skew'],
            'z_kurtosis': z_stats['kurtosis'],
            'z_autocorr_lag1': z_autocorr_lag1,
            'z_autocorr_lag2': z_autocorr_lag2,

            # Longest run statistics
            'run_mean': run_stats['mean'],
            'run_var': run_stats['var'],
            'run_min': run_stats['min'],
            'run_max': run_stats['max'],
            'run_skew': run_stats['skew'],
            'run_kurtosis': run_stats['kurtosis'],

            # Frequency statistics
            'freq_mean': freq_stats['mean'],
            'freq_var': freq_stats['var'],
            'freq_min': freq_stats['min'],
            'freq_max': freq_stats['max'],
            'freq_skew': freq_stats['skew'],
            'freq_kurtosis': freq_stats['kurtosis'],
        }

        # Add raw window values
        for i, z in enumerate(z_scores):
            result[f'z_score_{i}'] = z
        for i, run in enumerate(longest_runs):
            result[f'run_{i}'] = run
        for i, freq in enumerate(frequencies):
            result[f'freq_{i}'] = freq

        return result

    def process_file(self, input_file: str, output_file: str) -> pd.DataFrame:
        """
        Process a combined file with rolling window analysis.

        Args:
            input_file: Input CSV path
            output_file: Output CSV path

        Returns:
            DataFrame with rolling window features
        """
        print(f"Processing: {input_file}")

        df = pd.read_csv(input_file)
        results = []

        for _, row in df.iterrows():
            sequence = str(row['0_1_sequence'])
            result = self.process_sequence(
                sequence=sequence,
                global_z=row.get('z_score', 0.0),
                label=row['label'],
                row_id=row['id']
            )
            results.append(result)

        # Create output dataframe
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_file, index=False)

        print(f"✓ Saved to: {output_file} ({len(output_df)} rows)")
        return output_df

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Process all combined files in directory.

        Args:
            input_dir: Input directory with combined files
            output_dir: Output directory for rolling window results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Files to process
        files = [
            'combined_D1.csv', 'combined_D2.csv', 'combined_D3.csv',
            'combined_D4.csv', 'combined_D5.csv', 'combined_D6.csv',
            'combined_D7.csv', 'combined_D8.csv', 'combined_original_zscore.csv'
        ]

        for filename in files:
            input_file = input_path / filename
            output_file = output_path / filename.replace('combined_', 'rolling_')

            if input_file.exists():
                self.process_file(str(input_file), str(output_file))
            else:
                print(f"Warning: {input_file} not found")

        print("\n✅ Rolling window analysis complete!")