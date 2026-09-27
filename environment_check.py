import sys
import platform

print("=" * 80)
print("ENVIRONMENT CHECK")
print("=" * 80)

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Platform:", platform.platform())

print("\n" + "=" * 80)
print("LIBRARY CHECK")
print("=" * 80)

import numpy as np
import pandas as pd
import sklearn
import matplotlib
import torch
import transformers
import datasets
import sentence_transformers
import chromadb
import lime
import streamlit
import fastapi
import pydantic
import uvicorn
import joblib

print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("chromadb:", chromadb.__version__)
print("lime:", lime.__version__ if hasattr(lime, "__version__") else "installed")
print("streamlit:", streamlit.__version__)
print("fastapi:", fastapi.__version__)
print("pydantic:", pydantic.__version__)
print("uvicorn:", uvicorn.__version__)
print("joblib:", joblib.__version__)

print("\n" + "=" * 80)
print("GPU CHECK")
print("=" * 80)

print("CUDA available:", torch.cuda.is_available())
print("CUDA version used by torch:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("GPU not visible to PyTorch.")

print("\nSetup validation completed.")