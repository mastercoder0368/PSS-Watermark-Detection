"""
Helper Functions
Common utilities used across the PSS watermark detection pipeline.
"""

import os
import json
import yaml
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR)

    Returns:
        Logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"pss_watermark_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger('pss_watermark')
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def ensure_directories(paths: List[str]) -> None:
    """
    Ensure that directories exist.

    Args:
        paths: List of directory paths
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """
    Get the available device (cuda or cpu).

    Returns:
        Device string
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary')
    }

    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)

    return metrics


def save_checkpoint(state: Dict, checkpoint_dir: str,
                    filename: str = "checkpoint.pkl") -> str:
    """
    Save checkpoint for resuming processing.

    Args:
        state: State dictionary to save
        checkpoint_dir: Directory for checkpoints
        filename: Checkpoint filename

    Returns:
        Path to saved checkpoint
    """
    import pickle

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_dir) / filename

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state, f)

    return str(checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """
    Load checkpoint for resuming processing.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        State dictionary or None if not found
    """
    import pickle

    if not Path(checkpoint_path).exists():
        return None

    try:
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        return state
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def merge_csv_files(file_list: List[str], output_file: str,
                    on_column: Optional[str] = None) -> pd.DataFrame:
    """
    Merge multiple CSV files.

    Args:
        file_list: List of CSV file paths
        output_file: Output file path
        on_column: Column to merge on (if None, concatenate)

    Returns:
        Merged DataFrame
    """
    dfs = []

    for file_path in file_list:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"Loaded {file_path}: {len(df)} rows")

    if not dfs:
        print("No files to merge")
        return pd.DataFrame()

    if on_column:
        # Merge on specific column
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on=on_column, how='outer')
    else:
        # Concatenate
        merged = pd.concat(dfs, ignore_index=True)

    merged.to_csv(output_file, index=False)
    print(f"Saved merged file to {output_file}: {len(merged)} rows")

    return merged


def validate_dataset(data_dir: str) -> Dict[str, Any]:
    """
    Validate that all required dataset files exist.

    Args:
        data_dir: Data directory path

    Returns:
        Validation results dictionary
    """
    data_path = Path(data_dir)

    required_files = {
        'human_text.csv': 'Human text dataset',
        'ai_input_text.csv': 'AI input text dataset',
        'ai_watermarked_text.csv': 'AI watermarked text',
        'ai_watermark_bits.csv': 'AI watermark bits'
    }

    validation_results = {
        'valid': True,
        'missing_files': [],
        'file_info': {}
    }

    for filename, description in required_files.items():
        filepath = data_path / filename

        if filepath.exists():
            df = pd.read_csv(filepath)
            validation_results['file_info'][filename] = {
                'exists': True,
                'rows': len(df),
                'columns': list(df.columns),
                'description': description
            }
        else:
            validation_results['valid'] = False
            validation_results['missing_files'].append(filename)
            validation_results['file_info'][filename] = {
                'exists': False,
                'description': description
            }

    return validation_results


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory usage stats
    """
    import psutil

    process = psutil.Process()
    memory_info = process.memory_info()

    memory_stats = {
        'rss_gb': memory_info.rss / 1e9,  # Resident Set Size
        'vms_gb': memory_info.vms / 1e9,  # Virtual Memory Size
        'percent': process.memory_percent()
    }

    if torch.cuda.is_available():
        memory_stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
        memory_stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1e9

    return memory_stats


def print_experiment_summary(results_df: pd.DataFrame) -> None:
    """
    Print a formatted summary of experiment results.

    Args:
        results_df: DataFrame with experiment results
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    for _, row in results_df.iterrows():
        print(f"\n{row['Experiment']}:")
        print(f"  Accuracy: {row['Accuracy (%)']}")
        print(f"  AUC:      {row['AUC']}")
        print(f"  F1:       {row['F1']:.4f}")
        print(f"  Precision: {row['Precision']:.4f}")
        print(f"  Recall:    {row['Recall']:.4f}")

    print("\n" + "=" * 80)

    # Find best performing experiment
    results_df['Accuracy_num'] = results_df['Accuracy (%)'].str.rstrip('%').astype(float)
    best_idx = results_df['Accuracy_num'].idxmax()
    best_exp = results_df.loc[best_idx]

    print(f"BEST PERFORMING: {best_exp['Experiment']}")
    print(f"  Accuracy: {best_exp['Accuracy (%)']}")
    print("=" * 80)


def cleanup_temp_files(temp_dir: str = "temp") -> None:
    """
    Clean up temporary files.

    Args:
        temp_dir: Temporary directory path
    """
    temp_path = Path(temp_dir)

    if temp_path.exists():
        import shutil
        shutil.rmtree(temp_path)
        print(f"Cleaned up temporary files in {temp_dir}")


def get_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        MD5 hash string
    """
    import hashlib

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()
