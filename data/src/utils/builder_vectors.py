import numpy as np
from thirdai import dataset


class __BuilderVector__:
    """Builder vector interface. 
    Only to be used internally, so it is private (surrounded by __).
    A builder vector is a data structure for composing features
    from different blocks into a single vector.
    """

    def addSingleFeature(self, start_dim: int, value: float) -> None:
        """Sets the start_dim-th dimension of the vector to value."""
        return

    def addSparseFeatures(self, indices: np.ndarray, values: np.ndarray) -> None:
        """Sets the values at the given indices to the given values."""
        return

    def addDenseFeatures(self, start_dim: int, values: np.ndarray) -> None:
        """Appends a dense array of features to the vector starting at start_dim."""
        return

    def to_bolt_vector(self) -> dataset.BoltVector:
        """Converts the vector to the fixed-sized, sparsity-agnostic bolt vector."""
        return


class __SparseBuilderVector__(__BuilderVector__):
    """A concrete implementation of BuilderVector for sparse vectors."""

    def __init__(self) -> None:
        """Constructor.
        A sparse vector is implemented as lists of indices and corresponding values.
        """
        self._indices = []
        self._values = []

    def __str__(self):
        """Returns a string representation for printing."""
        return (
            "["
            + ", ".join(
                f"({idx}, {val})" for idx, val in zip(self._indices, self._values)
            )
            + "]"
        )

    def __repr__(self):
        """Returns a string representation."""
        return self.__str__()

    def addSingleFeature(self, start_dim: int, value: float) -> None:
        """Sets the start_dim-th dimension of the vector to value."""
        self._indices.append(start_dim)
        self._values.append(value)

    def addSparseFeatures(self, indices: np.ndarray, values: np.ndarray) -> None:
        """Sets the values at the given indices to the given values."""
        self._indices.extend(indices.tolist())
        self._values.extend(values.tolist())

    def addDenseFeatures(self, start_dim: int, values: np.ndarray) -> None:
        """Appends a dense array of features to the vector starting at start_dim."""
        self._indices.extend(np.arange(start_dim, start_dim + values.shape[0]))
        self._values.extend(values.tolist())

    def to_bolt_vector(self) -> dataset.BoltVector:
        """Converts the vector to the fixed-sized, sparsity-agnostic bolt vector."""
        sorted_lists = sorted(zip(self._indices, self._values))

        # Sort the arrays and throw error if there is a duplicate.
        # TODO(Geordie): It's a nice check but is this expensive? Is it necessary?
        self._indices = [idx for idx, _ in sorted_lists]
        self._values = [val for _, val in sorted_lists]
        last_idx = -1
        for idx in self._indices:
            if idx != last_idx:
                last_idx = idx
            else:
                raise RuntimeError("Found produced duplicate entries for the same"
                    + "position in the sparse vector")

        return dataset.make_sparse_vector(self._indices, self._values)


class __DenseBuilderVector__(__BuilderVector__):
    """A concrete implementation of __BuilderVector__ for dense vectors.
    
    Note that the dense builder vector expects that features are appended 
    in order (lower vector offset followed by higher vector offset) 
    and contiguously (the end of the vector section occupied by a feature
    marks the start of the vector section occupied by another feature).

    In effect, the first call to addDenseFeatures() / addSingleFeature()
    must have start_dim = 0, and the next call to these methods must have
    start_dim = the dimension of the previously added feature, and so on.
    """

    def __init__(self):
        """Constructor.
        A dense vector is just a list of floats.
        """
        self._values = []

    def __str__(self):
        """Returns a string representation for printing."""
        return "[" + ", ".join([str(val) for val in self._values]) + "]"

    def __repr__(self):
        """Returns a string representation."""
        return self.__str__()

    def addSingleFeature(self, start_dim: int, value: float) -> None:
        """Sets the start_dim-th dimension of the vector to value."""
        assert len(self._values) == start_dim
        self._values.append(value)

    def addSparseFeatures(self, indices: np.ndarray, values: np.ndarray) -> None:
        """Throws an error since dense features should not accept sparse features."""
        raise RuntimeError(
            "DenseBuilderVector: Attempted to add sparse features. Use SparseBuilderVector instead."
        )

    def addDenseFeatures(self, start_dim: int, values: np.ndarray) -> None:
        """Appends a dense array of features to the vector starting at start_dim."""
        assert len(self._values) == start_dim
        self._values.extend(values.tolist())

    def to_bolt_vector(self) -> dataset.BoltVector:
        """Converts the vector to the fixed-sized, sparsity-agnostic bolt vector."""
        return dataset.make_dense_vector(self._values)
