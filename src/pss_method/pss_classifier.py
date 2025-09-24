"""
PSS Classifier
Implements PSS method with XGBoost classification.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             confusion_matrix, precision_recall_fscore_support)


class PSSClassifier:
    """PSS classification with local z-scores and static features."""

    def __init__(self, rolling_dir: str, output_dir: str, exp_config: dict):
        """
        Initialize PSS classifier.

        Args:
            rolling_dir: Directory with rolling window results
            output_dir: Output directory for PSS results
            exp_config: Experiment configuration
        """
        self.rolling_dir = Path(rolling_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.exp_config = exp_config

        # Static features from configuration
        self.static_features = exp_config['pss_method']['static_features']

        # Classifier parameters
        self.clf_params = exp_config['classifier']

        # Load all depth dataframes
        self.depth_dfs = self._load_depth_files()

    def _load_depth_files(self) -> Dict[int, pd.DataFrame]:
        """
        Load all depth DataFrames.

        Returns:
            Dictionary mapping depth to DataFrame
        """
        depth_files = [
            'rolling_D1.csv', 'rolling_D2.csv', 'rolling_D3.csv',
            'rolling_D4.csv', 'rolling_D5.csv', 'rolling_D6.csv',
            'rolling_D7.csv', 'rolling_D8.csv'
        ]

        depth_dfs = {}

        for depth, filename in enumerate(depth_files, start=1):
            filepath = self.rolling_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                depth_dfs[depth] = df
                print(f"✓ Loaded {filename}: {len(df)} rows")
            else:
                print(f"Warning: {filepath} not found")

        return depth_dfs

    def compute_pss_matrix(self, dfs: List[pd.DataFrame],
                           n_windows: int) -> np.ndarray:
        """
        Compute PSS (standard deviation across depths) for z-scores.

        Args:
            dfs: List of DataFrames for different depths
            n_windows: Number of windows to use

        Returns:
            PSS matrix (n_samples × n_windows)
        """
        arrays = []

        for df in dfs:
            # Find z_score columns
            z_cols = [c for c in df.columns if c.startswith('z_score_')]
            # Take only first n_windows
            arrays.append(df[z_cols[:n_windows]].to_numpy())

        # Stack into shape (n_samples, n_windows, n_depths)
        stack = np.stack(arrays, axis=2)

        # Compute standard deviation across depth axis
        return np.std(stack, axis=2)

    def run_experiment(self, experiment_name: str,
                       depths: List[int]) -> Dict:
        """
        Run a single PSS experiment.

        Args:
            experiment_name: Name of experiment
            depths: List of depths to use

        Returns:
            Dictionary with results
        """
        print(f"\nRunning experiment: {experiment_name}")

        # Select dataframes for specified depths
        dfs = [self.depth_dfs[d] for d in depths if d in self.depth_dfs]

        if len(dfs) < 2:
            print(f"Warning: Need at least 2 depths for PSS, got {len(dfs)}")
            return None

        # Find minimum number of windows across depths
        z_cols_per_depth = [
            [c for c in df.columns if c.startswith('z_score_')]
            for df in dfs
        ]
        n_windows = min(len(cols) for cols in z_cols_per_depth)

        # Compute PSS matrix
        pss_matrix = self.compute_pss_matrix(dfs, n_windows)
        pss_df = pd.DataFrame(
            pss_matrix,
            columns=[f'pss_win{i + 1}' for i in range(n_windows)]
        )

        # Get metadata and static features from first depth
        meta = dfs[0][['id', 'label']].reset_index(drop=True)
        static_df = dfs[0][self.static_features].reset_index(drop=True)

        # Combine all features
        full_df = pd.concat([meta, pss_df, static_df], axis=1)
        full_df = full_df.fillna(0.0)  # Fill NaN with zeros

        # Train/test split
        X = full_df.drop(columns=['id', 'label'])
        y = full_df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.exp_config['training']['test_size'],
            stratify=y,
            random_state=self.exp_config['training']['random_seed']
        )

        # Train XGBoost
        clf = XGBClassifier(
            n_estimators=self.clf_params['n_estimators'],
            max_depth=self.clf_params['max_depth'],
            learning_rate=self.clf_params['learning_rate'],
            subsample=self.clf_params['subsample'],
            colsample_bytree=self.clf_params['colsample_bytree'],
            objective=self.clf_params['objective'],
            eval_metric=self.clf_params['eval_metric'],
            random_state=self.clf_params['random_state'],
            n_jobs=self.clf_params['n_jobs'],
            use_label_encoder=False
        )

        clf.fit(X_train, y_train)

        # Evaluate
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average='binary', pos_label=1
        )

        result = {
            'Experiment': experiment_name,
            'Accuracy (%)': f"{accuracy * 100:.2f}",
            'AUC': f"{auc:.3f}",
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4),
            'ConfusionMatrix': cm.tolist()
        }

        print(f"  Accuracy: {accuracy * 100:.2f}%")
        print(f"  AUC: {auc:.3f}")
        print(f"  F1: {f1:.3f}")

        return result

    def run_all_experiments(self) -> pd.DataFrame:
        """
        Run all experiments defined in configuration.

        Returns:
            DataFrame with all results
        """
        results = []

        # Standard experiments
        if 'depth_experiments' in self.exp_config:
            for exp in self.exp_config['depth_experiments']:
                result = self.run_experiment(exp['name'], exp['depths'])
                if result:
                    results.append(result)

        # Alternative experiments
        if 'alternative_depth_experiments' in self.exp_config:
            for exp in self.exp_config['alternative_depth_experiments']:
                result = self.run_experiment(exp['name'], exp['depths'])
                if result:
                    results.append(result)

        # Create results dataframe
        results_df = pd.DataFrame(results)

        # Save results
        output_path = self.output_dir / 'pss_results.csv'
        results_df.to_csv(output_path, index=False)

        print(f"\n✅ All experiments complete!")
        print(f"Results saved to: {output_path}")

        return results_df
