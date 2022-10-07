from .circular import Circular
from .gloo import Gloo
from .linear import Linear

AVAILABLE_METHODS = {"circular": Circular, "linear": Linear, "gloo": Gloo}
