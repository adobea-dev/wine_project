from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
readme = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="wine-quality-project",
    version="1.0.0",
    description="Wine Quality multi-class classification (RF + feature selection + pipelines)",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="adobea-dev",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "imbalanced-learn>=0.12",
        "scipy>=1.11",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "shap>=0.43",
        "flask>=2.3",
        "joblib>=1.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)