from setuptools import setup, find_packages

setup(
    name="f1-race-predictor",
    version="1.0.0",
    description="F1 Race Position Prediction using Deep Learning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "torch>=2.0.1",
        "scikit-learn>=1.3.0",
        "fastapi>=0.100.0",
        "mlflow>=2.5.0",
    ],
    python_requires=">=3.9",
)
