"""Task adapters for LEKA."""

from .pytorch_task import PyTorchTask, load_pytorch_task

__all__ = ["PyTorchTask", "load_pytorch_task"]
