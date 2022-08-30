from .linear import LinearCommunication as Linear
from .circular import CircularCommunication as Circular

AVAILABLE_METHODS = {
    "circular": Circular,
    "linear": Linear,
}
