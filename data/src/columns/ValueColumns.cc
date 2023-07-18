#include "ValueColumns.h"
#include "ColumnUtils.h"

namespace thirdai::data {

template <typename T>
void ValueColumnImpl<T>::shuffle(const std::vector<size_t>& permutation) {
  shuffleVector(_data, permutation);
}

template <typename T>
std::shared_ptr<Column> ValueColumnImpl<T>::concat(
    std::shared_ptr<Column>&& other) {
  if (_dimension != other->dimension()) {
    throw std::invalid_argument(
        "Can only concatenate columns with the same dimension.");
  }

  auto other_concrete = std::dynamic_pointer_cast<ValueColumnImpl<T>>(other);
  if (!other_concrete) {
    throw std::invalid_argument(
        "Can only concatenate value columns of the same type.");
  }

  auto new_data =
      concatVectors(std::move(_data), std::move(other_concrete->_data));

  auto new_column = std::shared_ptr<ValueColumnImpl>(
      new ValueColumnImpl<T>(std::move(new_data), _dimension));

  _dimension.reset();
  other_concrete->_dimension.reset();

  return new_column;
}

template class ValueColumnImpl<std::string>;
template class ValueColumnImpl<uint32_t>;
template class ValueColumnImpl<float>;
template class ValueColumnImpl<int64_t>;

}  // namespace thirdai::data