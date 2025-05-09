import os
from typing import Optional, Union

def build_model_path(base_path: str, suffix: str) -> str:
    """
    Utility to insert a suffix **before** the file extension.

    Example:
        build_model_path("model.pth", "_best") -> "model_best.pth"
    """
    root, ext = os.path.splitext(base_path)
    return f"{root}{suffix}{ext}"