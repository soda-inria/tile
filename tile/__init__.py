"""TILE: Table-Interfaced LLM Embeddings."""

from importlib.metadata import PackageNotFoundError, version

from tile.embeddings import Embeddings
from tile.learners import LEARNERS
from tile.llms import DEFAULT_LLM, LLMS
from tile.model import TILE

try:
    __version__ = version("tile")
except PackageNotFoundError:
    # Run from a source tree that was not installed (`pip install -e .`)
    __version__ = "unknown"

__all__ = ["DEFAULT_LLM", "Embeddings", "LEARNERS", "LLMS", "TILE", "__version__"]
