#pragma once

#include <dataset/src/data_pipeline/Column.h>
#include <stdexcept>

namespace thirdai::dataset {

template <typename T>
class VectorValueColumn final : public ValueColumn<T> {
 public:
  // This uses SFINAE to disable the folowing constructor if T is not a certain
  // type. https://en.cppreference.com/w/cpp/types/enable_if
  template <typename U = T,
            std::enable_if_t<std::is_same<U, uint32_t>::value, bool> = true>
  explicit VectorValueColumn(std::vector<uint32_t> data, uint32_t dim)
      : _data(std::move(data)), _dim(dim) {
    for (uint32_t index : _data) {
      if (index >= _dim) {
        throw std::out_of_range("Cannot have index " + std::to_string(index) +
                                " in VectorIntegerValueColumn of dimension " +
                                std::to_string(_dim) + ".");
      }
    }
  }

  // This uses SFINAE to disable the folowing constructor if T is not a certain
  // type. https://en.cppreference.com/w/cpp/types/enable_if
  template <typename U = T,
            std::enable_if_t<std::is_same<U, float>::value, bool> = true>
  explicit VectorValueColumn(std::vector<float> data)
      : _data(std::move(data)), _dim(1) {}

  uint64_t numRows() const final { return _data.size(); }

  uint32_t dim() const final { return _dim; }

  const T& operator[](uint64_t n) const final { return _data.at(n); }

 private:
  std::vector<T> _data;
  uint32_t _dim;
};

}  // namespace thirdai::dataset