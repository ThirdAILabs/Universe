from .circular import Circular
from .linear import Linear
from .gloo import Gloo

AVAILABLE_METHODS = {
    "circular": Circular,
    "linear": Linear,
    "gloo": Gloo
}
