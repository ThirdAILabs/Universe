from .linear import Linear
from .circular import Circular
from .tree import Tree

AVAILABLE_METHODS = {
    "circular": Circular,
    "linear": Linear,
    "tree": Tree,
}
