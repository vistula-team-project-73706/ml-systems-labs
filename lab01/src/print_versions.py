import sys
import importlib.metadata


PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "torch",
    "torchvision",
    "torchinfo",
    "thop",
    "onnx",
    "onnxruntime",
    "mlflow",
    "memory-profiler",
    "psutil",
    "codecarbon",
    "fastapi",
    "uvicorn",
    "pytest",
    "httpx",
    "locust",
    "requests",
    "pyarrow",
    "joblib",
    "tqdm",
]


print(f"Python: {sys.version}")

for package in PACKAGES:
    try:
        version = importlib.metadata.version(package)
        print(f"{package}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
