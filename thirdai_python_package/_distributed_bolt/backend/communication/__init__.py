from .linear import Linear
from .circular import Circular
from .gloo import GlooBackend

AVAILABLE_METHODS = {
    "circular": Circular,
    "linear": Linear,
    "gloo": GlooBackend
}
