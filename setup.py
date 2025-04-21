from setuptools import setup, find_packages

setup(
    name="portfolio-rl-mlops",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "yfinance>=0.1.70",
        "gymnasium>=0.28.1",
        "stable-baselines3>=2.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pylint>=2.8.3",
            "black>=21.6b0",
            "isort>=5.9.2",
        ],
        "mlops": [
            "mlflow>=2.0.0",
            "dvc>=2.10.0",
            "dvc-s3>=2.10.0",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A reinforcement learning approach for portfolio optimization with MLOps",
)