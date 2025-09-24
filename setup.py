"""
Setup script for PSS Watermark Detection package.
"""

import os
from setuptools import setup, find_packages


# Read README for long description
def read_file(filename):
    """Read file and return contents."""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


# Get version from __init__.py
def get_version():
    """Extract version from __init__.py."""
    init_path = os.path.join('src', '__init__.py')
    with open(init_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '1.0.0'


# Core dependencies
REQUIRED = [
    'numpy>=1.21.0',
    'pandas>=1.3.0',
    'torch>=1.9.0',
    'transformers>=4.30.0',
    'scipy>=1.7.0',
    'scikit-learn>=1.0.0',
    'xgboost>=1.5.0',
    'tqdm>=4.62.0',
    'pyyaml>=5.4.0',
    'huggingface-hub>=0.16.0',
    'tokenizers>=0.13.0',
    'nltk>=3.7',
    'psutil>=5.8.0',
]

# Optional dependencies for different use cases
EXTRAS = {
    'paraphrasing': [
        'llama-cpp-python>=0.2.0',
    ],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'mypy>=0.950',
        'isort>=5.10.0',
        'pre-commit>=2.17.0',
    ],
    'docs': [
        'sphinx>=4.5.0',
        'sphinx-rtd-theme>=1.0.0',
        'sphinx-autodoc-typehints>=1.18.0',
    ],
    'kaggle': [
        'kaggle>=1.5.0',
    ],
    'notebook': [
        'jupyter>=1.0.0',
        'ipykernel>=6.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
    ],
}

# All optional dependencies
EXTRAS['all'] = sum(EXTRAS.values(), [])

setup(
    name='pss-watermark-detection',
    version=get_version(),
    author='Your Name',
    author_email='your.email@example.com',
    description='PSS method for robust watermark detection in paraphrased texts',
    long_description=read_file('README.md') if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/mastercoder0368/pss-watermark-detection',
    license='MIT',

    # Package configuration
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'examples']),
    package_dir={'': '.'},
    include_package_data=True,
    python_requires='>=3.8',

    # Dependencies
    install_requires=REQUIRED,
    extras_require=EXTRAS,

    # Entry points for command line scripts
    entry_points={
        'console_scripts': [
            'pss-pipeline=scripts.run_pipeline:main',
            'pss-dataset=scripts.run_dataset_creation:main',
            'pss-watermark=scripts.run_watermarking:main',
            'pss-paraphrase=scripts.run_paraphrasing:main',
            'pss-detect=scripts.run_detection:main',
            'pss-analyze=scripts.run_pss_analysis:main',
        ],
    },

    # Classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Environment :: GPU :: NVIDIA CUDA',
    ],

    # Keywords for discovery
    keywords=[
        'watermarking',
        'text-detection',
        'paraphrase',
        'machine-learning',
        'nlp',
        'natural-language-processing',
        'ai-safety',
        'llm',
        'large-language-models',
        'text-generation',
    ],

    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/mastercoder0368/pss-watermark-detection/issues',
        'Source': 'https://github.com/mastercoder0368/pss-watermark-detection',
        'Documentation': 'https://pss-watermark-detection.readthedocs.io/',
        'Paper': '',#'https://arxiv.org/abs/your_paper_id',
    },

    # Data files to include
    package_data={
        'src': ['../configs/*.yaml'],
    },

    # Additional files to include
    data_files=[
        ('configs', ['configs/experiment_config.yaml', 'configs/model_config.yaml']),
    ],

    # Minimum versions for critical packages
    dependency_links=[],

    # Testing
    test_suite='tests',
    tests_require=[
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
    ],

    # Build configuration
    zip_safe=False,
)
