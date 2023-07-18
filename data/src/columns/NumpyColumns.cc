#include "NumpyColumns.h"
#include <optional>

namespace thirdai::data {

template <typename T>
void checkArrayis1D(const NumpyArray<T>& array) {
  if (array.ndim() != 1 && (array.ndim() != 2 || array.shape(1) != 1)) {
    throw std::invalid_argument(
        "Can only construct NumpyValueColumn from 1D numpy array.");
  }
}

template <typename T>
void checkArrayIs2D(const NumpyArray<T>& array) {
  if (array.ndim() != 2) {
    throw std::invalid_argument(
        "Can only construct NumpyArrayColumn from 2D numpy array.");
  }
}

void verifySparseArrayIndices(const NumpyArray<uint32_t>& array, uint32_t dim) {
  const uint32_t* data = array.data();
  for (uint32_t i = 0; i < array.size(); i++) {
    if (data[i] >= dim) {
      throw std::out_of_range("Cannot have index " + std::to_string(data[i]) +
                              " in Sparse Numpy Column of dimension " +
                              std::to_string(dim) + ".");
    }
  }
}

NumpyTokenColumn::NumpyTokenColumn(const NumpyArray<uint32_t>& array,
                                   std::optional<size_t> dim) {
  checkArrayis1D(array);

  if (dim) {
    verifySparseArrayIndices(array, *dim);
  }

  _dimension =
      dim ? std::make_optional<ColumnDimension>(*dim, false) : std::nullopt;
  _buffer_info = array.request();
}

NumpyDecimalColumn::NumpyDecimalColumn(const NumpyArray<float>& array) {
  checkArrayis1D(array);

  _dimension = ColumnDimension(1, true);
  _buffer_info = array.request();
}

NumpyTokenArrayColumn::NumpyTokenArrayColumn(const NumpyArray<uint32_t>& array,
                                             std::optional<size_t> dim) {
  checkArrayIs2D(array);

  if (dim) {
    verifySparseArrayIndices(array, *dim);
  }

  _dimension =
      dim ? std::make_optional<ColumnDimension>(*dim, false) : std::nullopt;
  _buffer_info = array.request();
}

NumpyDecimalArrayColumn::NumpyDecimalArrayColumn(
    const NumpyArray<float>& array) {
  checkArrayIs2D(array);

  _dimension = ColumnDimension(array.shape(1), true);
  _buffer_info = array.request();
}

template class NumpyArrayColumn<uint32_t>;
template class NumpyArrayColumn<float>;

}  // namespace thirdai::data