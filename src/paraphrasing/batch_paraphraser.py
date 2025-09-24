"""
Batch Paraphraser
Performs iterative paraphrasing using Mistral-7B model.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, List
import re

try:
    from llama_cpp import Llama
except ImportError:
    print("Warning: llama-cpp-python not installed. Please install with CUDA support.")
    print("pip install llama-cpp-python --force-reinstall --no-cache-dir")


class BatchParaphraser:
    """Performs batch paraphrasing with multiple iterations."""

    def __init__(self, model_config: dict = None, paraphrase_params: dict = None,
                 iterations: int = 8, batch_size: int = 10):
        """
        Initialize batch paraphraser.

        Args:
            model_config: Model configuration dictionary
            paraphrase_params: Paraphrasing parameters
            iterations: Number of paraphrasing iterations
            batch_size: Batch size for processing
        """
        # Default configurations
        if model_config is None:
            model_config = {
                'repo_id': 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
                'filename': 'mistral-7b-instruct-v0.2.Q4_K_M.gguf',
                'n_ctx': 4096,
                'n_gpu_layers': -1
            }

        if paraphrase_params is None:
            paraphrase_params = {
                'temperature': 0.8,
                'top_p': 0.95,
                'max_tokens_multiplier': 1.5,
                'stop_tokens': ['<s>', '</s>'],
                'word_count_tolerance': 50
            }

        self.model_config = model_config
        self.paraphrase_params = paraphrase_params
        self.iterations = iterations
        self.batch_size = batch_size
        self.llm = None
        self.save_frequency = 50

    def download_model(self, model_dir: str = "models") -> str:
        """
        Download model from HuggingFace if not present.

        Args:
            model_dir: Directory to save model

        Returns:
            Path to model file
        """
        from huggingface_hub import hf_hub_download

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Check if model already exists
        model_file = model_path / self.model_config['filename']
        if model_file.exists():
            print(f"Model already exists: {model_file}")
            return str(model_file)

        print(f"Downloading {self.model_config['filename']}...")

        # Set cache directory
        cache_dir = model_path / "cache"
        cache_dir.mkdir(exist_ok=True)

        # Download model
        downloaded_path = hf_hub_download(
            repo_id=self.model_config['repo_id'],
            filename=self.model_config['filename'],
            cache_dir=str(cache_dir)
        )

        # Copy to model directory
        import shutil
        shutil.copy2(downloaded_path, model_file)
        print(f"✓ Model downloaded to: {model_file}")

        return str(model_file)

    def load_model(self, model_path: str) -> None:
        """
        Load the GGUF model.

        Args:
            model_path: Path to GGUF model file
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model from {model_path}...")
        start_time = time.time()

        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=self.model_config.get('n_ctx', 4096),
                n_gpu_layers=self.model_config.get('n_gpu_layers', -1)
            )
            print(f"✓ Model loaded in {time.time() - start_time:.2f} seconds")

            # Verify GPU usage
            if hasattr(self.llm, 'n_gpu_layers'):
                print(f"GPU layers: {self.llm.n_gpu_layers}")
            print(f"Context size: {self.llm.n_ctx}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def create_paraphrase_prompt(self, text: str, min_words: int, max_words: int) -> str:
        """
        Create paraphrase prompt for Mistral format.

        Args:
            text: Input text to paraphrase
            min_words: Minimum word count
            max_words: Maximum word count

        Returns:
            Formatted prompt
        """
        prompt = f"""<s>[INST] You are a paraphraser. Take the following text and rewrite it completely.

REQUIREMENTS:
- The paraphrased output MUST have more than {min_words} words
- Change all sentence structures and vocabulary
- Keep ALL information and details from the original
- Do not summarize or shorten
- Do not add new information
- Provide only the paraphrased output

INPUT TEXT:
{text}

OUTPUT: [/INST]"""
        return prompt

    def clean_model_output(self, text: str) -> str:
        """
        Clean the model output.

        Args:
            text: Raw model output

        Returns:
            Cleaned text
        """
        # Remove "OUTPUT:" prefix if present
        cleaned = re.sub(r'^OUTPUT:\s*', '', text, flags=re.IGNORECASE)
        return cleaned.strip()

    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(str(text).split())

    def paraphrase_text(self, text: str, target_word_count: int) -> Tuple[str, int]:
        """
        Paraphrase a single text.

        Args:
            text: Input text
            target_word_count: Target word count

        Returns:
            Tuple of (paraphrased_text, actual_word_count)
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        min_words = target_word_count - self.paraphrase_params['word_count_tolerance']
        max_words = target_word_count + self.paraphrase_params['word_count_tolerance']

        # Calculate max tokens (approximate)
        max_tokens = int(max_words * self.paraphrase_params['max_tokens_multiplier'])

        # Create prompt
        prompt = self.create_paraphrase_prompt(text, min_words, max_words)

        try:
            # Generate paraphrase
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=self.paraphrase_params['temperature'],
                top_p=self.paraphrase_params['top_p'],
                stop=self.paraphrase_params['stop_tokens']
            )

            generated_text = output["choices"][0]["text"].strip()
            cleaned_text = self.clean_model_output(generated_text)
            word_count = self.count_words(cleaned_text)

            return cleaned_text, word_count

        except Exception as e:
            print(f"Error generating paraphrase: {e}")
            return f"ERROR: {str(e)}", 0

    def get_resume_info(self, output_csv: str) -> Tuple[int, int]:
        """
        Get resume information from existing output file.

        Args:
            output_csv: Path to output CSV

        Returns:
            Tuple of (iteration_start, row_start)
        """
        if not os.path.exists(output_csv):
            return 0, 0

        try:
            df = pd.read_csv(output_csv)

            # Find incomplete iterations
            paraphrase_cols = [col for col in df.columns if col.startswith('paraphrase_iter_')]

            if not paraphrase_cols:
                return 0, 0

            # Sort by iteration number
            paraphrase_cols.sort(key=lambda x: int(x.split('_')[-1]))

            for col_idx, col_name in enumerate(paraphrase_cols):
                if df[col_name].isna().any():
                    row_start = df[col_name].isna().idxmax()
                    return col_idx, row_start

            # All complete, start new iteration
            return len(paraphrase_cols), 0

        except Exception as e:
            print(f"Error reading existing file: {e}")
            return 0, 0

    def process_csv(self, input_csv: str, output_csv: str,
                    text_column: str = 'text_cleaned',
                    word_count_column: str = 'new_word_count',
                    model_path: str = None) -> pd.DataFrame:
        """
        Process a CSV file with multiple paraphrasing iterations.

        Args:
            input_csv: Input CSV path
            output_csv: Output CSV path
            text_column: Column containing text
            word_count_column: Column containing word counts
            model_path: Path to model file

        Returns:
            DataFrame with paraphrased texts
        """
        print(f"Loading CSV from {input_csv}")

        # Load input data
        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

        # Verify columns exist
        if text_column not in df.columns:
            print(f"Error: Column '{text_column}' not found")
            print(f"Available columns: {df.columns.tolist()}")
            # Try to find a suitable column
            for col in ['text', 'generated_text', 'text_cleaned']:
                if col in df.columns:
                    text_column = col
                    print(f"Using column: {text_column}")
                    break
            else:
                raise ValueError(f"No suitable text column found")

        if word_count_column not in df.columns:
            print(f"Warning: Column '{word_count_column}' not found, calculating from text")
            df[word_count_column] = df[text_column].apply(lambda x: len(str(x).split()))

        # Load model if needed
        if model_path and self.llm is None:
            self.load_model(model_path)
        elif self.llm is None:
            raise RuntimeError("No model loaded and no model_path provided")

        # Check for resume
        iteration_start, row_start = self.get_resume_info(output_csv)

        # Initialize or load output dataframe
        if os.path.exists(output_csv) and iteration_start > 0:
            output_df = pd.read_csv(output_csv)
            print(f"Resuming from iteration {iteration_start + 1}, row {row_start}")
        else:
            output_df = pd.DataFrame()
            output_df['original_text'] = df[text_column]
            output_df['original_word_count'] = df[word_count_column]

        # Process iterations
        for iter_idx in range(iteration_start, self.iterations):
            print(f"\n{'=' * 60}")
            print(f"Iteration {iter_idx + 1}/{self.iterations}")
            print(f"{'=' * 60}")

            # Determine source column
            if iter_idx == 0:
                source_col = 'original_text'
            else:
                source_col = f'paraphrase_iter_{iter_idx}'

            # Output columns
            target_col = f'paraphrase_iter_{iter_idx + 1}'
            word_count_col = f'word_count_iter_{iter_idx + 1}'

            # Ensure columns exist
            if target_col not in output_df.columns:
                output_df[target_col] = None
            if word_count_col not in output_df.columns:
                output_df[word_count_col] = None

            # Process texts
            total_rows = len(output_df)
            start_batch = (row_start // self.batch_size) * self.batch_size

            for batch_start in range(start_batch, total_rows, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_rows)
                print(f"Processing batch {batch_start}-{batch_end} of {total_rows}")

                for idx in tqdm(range(batch_start, batch_end)):
                    # Skip if already processed
                    if iter_idx == iteration_start and idx < row_start:
                        continue

                    if pd.isna(output_df.at[idx, target_col]):
                        input_text = output_df.at[idx, source_col]

                        if pd.isna(input_text):
                            print(f"Warning: Input at index {idx} is NaN")
                            continue

                        # Get target word count
                        if iter_idx == 0:
                            target_word_count = int(output_df.at[idx, 'original_word_count'])
                        else:
                            target_word_count = self.count_words(input_text)

                        # Clean input if from previous iteration
                        if iter_idx > 0:
                            input_text = self.clean_model_output(input_text)

                        # Paraphrase
                        paraphrased, word_count = self.paraphrase_text(
                            input_text, target_word_count
                        )

                        # Store results
                        output_df.at[idx, target_col] = paraphrased
                        output_df.at[idx, word_count_col] = word_count

                # Save progress
                if (batch_end % self.save_frequency == 0) or (batch_end == total_rows):
                    output_df.to_csv(output_csv, index=False)
                    print(f"Saved progress to {output_csv}")

            # Reset row_start for next iteration
            row_start = 0

        # Final save
        output_df.to_csv(output_csv, index=False)
        print(f"\n✓ All iterations complete! Results saved to {output_csv}")

        # Print summary
        print("\nWord Count Summary:")
        for i in range(1, self.iterations + 1):
            col = f'word_count_iter_{i}'
            if col in output_df.columns:
                avg = output_df[col].mean()
                print(f"  Iteration {i}: Average {avg:.1f} words")

        return output_df
