"""
PSS Watermark Detection
A comprehensive framework for detecting watermarks in paraphrased texts.

This package implements the Paraphrase Stability Score (PSS) method for
robust watermark detection in AI-generated texts, even after multiple
paraphrasing iterations.
"""

# Import main components for easier access
from .data_preparation import PG19DatasetCreator, TextLengthAdjuster
from .watermarking import (
    WatermarkBase,
    WatermarkProcessor,
    WatermarkLogitsProcessor,
    WatermarkDetector as BaseWatermarkDetector
)
from .paraphrasing import BatchParaphraser
from .detection import WatermarkDetector, FileSplitter
from .pss_method import (
    DataCombiner,
    RollingWindowAnalyzer,
    PSSClassifier
)
from .utils import (
    setup_logging,
    load_config,
    ensure_directories,
    get_device,
    calculate_metrics
)

# Define what gets imported with "from src import *"
__all__ = [
    # Version info
    '__version__',

    # Data preparation
    'PG19DatasetCreator',
    'TextLengthAdjuster',

    # Watermarking
    'WatermarkBase',
    'WatermarkProcessor',
    'WatermarkLogitsProcessor',
    'BaseWatermarkDetector',

    # Paraphrasing
    'BatchParaphraser',

    # Detection
    'WatermarkDetector',
    'FileSplitter',

    # PSS Method
    'DataCombiner',
    'RollingWindowAnalyzer',
    'PSSClassifier',

    # Utilities
    'setup_logging',
    'load_config',
    'ensure_directories',
    'get_device',
    'calculate_metrics',
]

# Package metadata
PACKAGE_INFO = {
    'name': 'pss-watermark-detection',
    'version': __version__,
    'description': 'PSS method for robust watermark detection in paraphrased texts',
    'author': __author__,
    'email': __email__,
    'license': 'MIT',
    'url': 'https://github.com/yourusername/pss-watermark-detection',
    'python_requires': '>=3.8',
    'keywords': [
        'watermarking',
        'text-detection',
        'paraphrase',
        'machine-learning',
        'nlp',
        'ai-safety'
    ]
}


def get_version():
    """Return the package version."""
    return __version__


def get_package_info():
    """Return complete package information."""
    return PACKAGE_INFO
