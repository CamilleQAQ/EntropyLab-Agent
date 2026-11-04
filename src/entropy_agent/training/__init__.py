"""Dataset-independent training, splitting, and evaluation utilities."""

from .pipeline import TrainingResult, train_dataset

__all__ = ["TrainingResult", "train_dataset"]
