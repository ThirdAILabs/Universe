import numpy as np
from thirdai import dataset


class BuilderVector:
    """Builder vector interface.
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


class SparseBuilderVector(BuilderVector):
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

        # Deduplicate entries by aggregating the values for the same index.
        # We use arrays and then sort and deduplicate instead of using hash maps
        # because hash maps are slow.
        real_size = -1
        last_idx = -1
        for iv in sorted_lists:
            if iv[0] != last_idx:
                real_size += 1
                last_idx = iv[0]

                self._indices[real_size] = iv[0]
                self._values[real_size] = iv[1]

            else:
                self._values[real_size] += iv[1]

        real_size += 1

        self._indices = self._indices[:real_size]
        self._values = self._values[:real_size]

        return dataset.make_sparse_vector(self._indices, self._values)


class DenseBuilderVector(BuilderVector):
    """A concrete implementation of BuilderVector for dense vectors."""

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
        if len(self._values > start_dim):
            raise RuntimeError(
                f"DenseBuilderVector: Adding feature at dimension {start_dim} but vector is already {len(self._values)} dimensions."
            )
        if len(self._values) < start_dim:
            self._values.extend([0 for _ in range(start_dim - len(self._values))])
        self._values.append(value)

    def addSparseFeatures(self, indices: np.ndarray, values: np.ndarray) -> None:
        """Throws an error since dense features should not accept sparse features."""
        raise RuntimeError(
            "DenseBuilderVector: Attempted to add sparse features. Use SparseBuilderVector instead."
        )

    def addDenseFeatures(self, start_dim: int, values: np.ndarray) -> None:
        """Appends a dense array of features to the vector starting at start_dim."""
        self._values.extend(values.tolist())

    def to_bolt_vector(self) -> dataset.BoltVector:
        """Converts the vector to the fixed-sized, sparsity-agnostic bolt vector."""
        return dataset.make_dense_vector(self._values)
