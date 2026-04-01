"""
Setup configuration for SensorDrift-MLOps package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sensor-drift-mlops",
    version="0.1.0",
    author="Satyan",
    author_email="satyan@example.com",
    description="Real-time wearable sensor drift detection with automated retraining",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sensor-drift-mlops",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "mlflow>=1.20.0",
        "fastapi>=0.75.0",
        "uvicorn>=0.17.0",
        "pydantic>=1.8.0",
        "dash>=2.0.0",
        "plotly>=5.0.0",
        "pytest>=6.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
)
