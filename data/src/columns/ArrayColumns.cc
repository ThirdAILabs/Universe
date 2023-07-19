#include "ArrayColumns.h"
#include "ColumnUtils.h"
#include <memory>

namespace thirdai::data {

template <typename T>
void ArrayColumn<T>::shuffle(const std::vector<size_t>& permutation) {
  _data = shuffleVector(std::move(_data), permutation);
}

template class ArrayColumn<uint32_t>;
template class ArrayColumn<float>;

template <typename T>
std::shared_ptr<Column> ArrayColumn<T>::concat(
    std::shared_ptr<Column>&& other) {
  if (_dimension != other->dimension()) {
    throw std::invalid_argument(
        "Can only concatenate columns with the same dimension.");
  }

  auto other_concrete = std::dynamic_pointer_cast<ArrayColumn<T>>(other);
  if (!other_concrete) {
    throw std::invalid_argument(
        "Can only concatenate value columns of the same type.");
  }

  auto new_data =
      concatVectors(std::move(_data), std::move(other_concrete->_data));

  auto new_column = std::shared_ptr<ArrayColumn>(
      new ArrayColumn<T>(std::move(new_data), _dimension));

  _dimension.reset();
  other_concrete->_dimension.reset();

  return new_column;
}

ArrayColumnPtr<uint32_t> makeTokenArrayColumn(
    std::vector<std::vector<uint32_t>>&& data, std::optional<size_t> dim) {
  if (dim) {
    for (const auto& row : data) {
      for (uint32_t index : row) {
        if (index >= *dim) {
          throw std::invalid_argument("Invalid index " + std::to_string(index) +
                                      " for TokenArrayColumn with dimension " +
                                      std::to_string(*dim) + ".");
        }
      }
    }
  }

  return ArrayColumnPtr<uint32_t>(
      new ArrayColumn<uint32_t>(std::move(data), ColumnDimension::sparse(dim)));
}

ArrayColumnPtr<float> makeDecimalArrayColumn(
    std::vector<std::vector<float>>&& data) {
  size_t dim = 0;
  if (!data.empty()) {
    dim = data.front().size();

    for (const auto& row : data) {
      if (row.size() != dim) {
        throw std::invalid_argument(
            "Expected consistent dimension in DecimalArrayColumn.");
      }
    }
  }

  return ArrayColumnPtr<float>(
      new ArrayColumn<float>(std::move(data), ColumnDimension::dense(dim)));
}

}  // namespace thirdai::data