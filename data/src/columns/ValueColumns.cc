#include "ValueColumns.h"
#include "ColumnUtils.h"

namespace thirdai::data {

template <typename T>
void ValueColumn<T>::shuffle(const std::vector<size_t>& permutation) {
  _data = shuffleVector(std::move(_data), permutation);
}

template <typename T>
std::shared_ptr<Column> ValueColumn<T>::concat(
    std::shared_ptr<Column>&& other) {
  if (_dimension != other->dimension()) {
    throw std::invalid_argument(
        "Can only concatenate columns with the same dimension.");
  }

  auto other_concrete = std::dynamic_pointer_cast<ValueColumn<T>>(other);
  if (!other_concrete) {
    throw std::invalid_argument(
        "Can only concatenate value columns of the same type.");
  }

  auto new_data =
      concatVectors(std::move(_data), std::move(other_concrete->_data));

  auto new_column = std::shared_ptr<ValueColumn>(
      new ValueColumn<T>(std::move(new_data), _dimension));

  _dimension.reset();
  other_concrete->_dimension.reset();

  return new_column;
}

ValueColumnPtr<uint32_t> makeTokenColumn(std::vector<uint32_t>&& data,
                                         std::optional<size_t> dim) {
  if (dim) {
    for (uint32_t index : data) {
      if (index >= *dim) {
        throw std::invalid_argument("Invalid index " + std::to_string(index) +
                                    " for TokenColumn with dimension " +
                                    std::to_string(*dim) + ".");
      }
    }
  }

  return ValueColumnPtr<uint32_t>(
      new ValueColumn<uint32_t>(std::move(data), ColumnDimension::sparse(dim)));
}

ValueColumnPtr<float> makeDecimalColumn(std::vector<float>&& data) {
  return ValueColumnPtr<float>(
      new ValueColumn<float>(std::move(data), ColumnDimension::dense(1)));
}

ValueColumnPtr<std::string> makeStringColumn(std::vector<std::string>&& data) {
  return ValueColumnPtr<std::string>(
      new ValueColumn<std::string>(std::move(data), std::nullopt));
}

ValueColumnPtr<int64_t> makeTimestampColumn(std::vector<int64_t>&& data) {
  return ValueColumnPtr<int64_t>(
      new ValueColumn<int64_t>(std::move(data), std::nullopt));
}

}  // namespace thirdai::data