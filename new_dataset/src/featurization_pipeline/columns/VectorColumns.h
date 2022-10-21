#pragma once

#include <_types/_uint32_t.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <sys/types.h>
#include <limits>
#include <stdexcept>

namespace thirdai::dataset {

template <typename T>
class VectorValueColumn final : public ValueColumn<T> {
 public:
  // This uses SFINAE to disable the folowing constructor if T is not a uint32_t
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

  // This uses SFINAE to disable the folowing constructor if T is not a float
  // type. https://en.cppreference.com/w/cpp/types/enable_if
  template <typename U = T,
            std::enable_if_t<std::is_same<U, float>::value, bool> = true>
  explicit VectorValueColumn(std::vector<float> data)
      : _data(std::move(data)), _dim(1) {}

  // This uses SFINAE to disable the folowing constructor if T is not a string
  // type. https://en.cppreference.com/w/cpp/types/enable_if
  template <typename U = T,
            std::enable_if_t<std::is_same<U, std::string>::value, bool> = true>
  explicit VectorValueColumn(std::vector<std::string> data)
      : _data(std::move(data)), _dim(1) {}

  uint64_t numRows() const final { return _data.size(); }

  std::optional<DimensionInfo> dimension() const final {
    if constexpr (std::is_same<T, uint32_t>::value ||
                  std::is_same<T, float>::value) {
      return {{_dim, std::is_same<T, float>::value}};
    }
    return std::nullopt;
  }

  const T& operator[](uint64_t n) const final { return _data.at(n); }

 private:
  std::vector<T> _data;
  uint32_t _dim;
};

template <typename T>
class VectorArrayColumn final : public ArrayColumn<T> {
  static_assert(std::is_same<T, uint32_t>::value ||
                    std::is_same<T, float>::value,
                "Only vectors of type uint32 or float32 can be used to "
                "construct columns.");

 public:
  // This uses SFINAE to disable the folowing constructor if T is not a uint32_t
  // type. https://en.cppreference.com/w/cpp/types/enable_if
  template <typename U = T,
            std::enable_if_t<std::is_same<U, uint32_t>::value, bool> = true>
  explicit VectorArrayColumn(std::vector<std::vector<uint32_t>> data,
                             uint32_t dim)
      : _data(std::move(data)), _dim(dim) {
    for (uint64_t row_index = 0; row_index < numRows(); row_index++) {
      for (uint32_t index : _data[row_index]) {
        if (index >= _dim) {
          throw std::out_of_range("Cannot have index " + std::to_string(index) +
                                  " in VectorSparseArrayColumn of dimension " +
                                  std::to_string(_dim) + ".");
        }
      }
    }
  }

  // This uses SFINAE to disable the folowing constructor if T is not a float
  // type. https://en.cppreference.com/w/cpp/types/enable_if
  template <typename U = T,
            std::enable_if_t<std::is_same<U, float>::value, bool> = true>
  explicit VectorArrayColumn(std::vector<std::vector<float>> data, uint32_t dim)
      : _data(std::move(data)), _dim(dim) {
    for (uint64_t row_index = 0; row_index < numRows(); row_index++) {
      if (_data[row_index].size() > _dim) {
        throw std::out_of_range("Cannot have vector of length " +
                                std::to_string(_data[row_index].size()) +
                                " in VectorDenseArrayColumn of dimension " +
                                std::to_string(_dim) + ".");
      }
    }
  }

  // This uses SFINAE to disable the folowing constructor if T is not a pair
  // type. https://en.cppreference.com/w/cpp/types/enable_if
  template <typename U = T,
            std::enable_if_t<std::is_same<U, std::pair<uint32_t, float>>::value,
                             bool> = true>
  explicit VectorArrayColumn(
      std::vector<std::vector<std::pair<uint32_t, float>>> data, uint32_t dim)
      : _data(std::move(data)), _dim(dim) {
    for (uint64_t row_index = 0; row_index < numRows(); row_index++) {
      for (auto [index, _] : _data[row_index]) {
        if (index >= _dim) {
          throw std::out_of_range(
              "Cannot have index " + std::to_string(index) +
              " in VectorIndexValueArrayColumn of dimension " +
              std::to_string(_dim) + ".");
        }
      }
    }
  }

  std::optional<DimensionInfo> dimension() const final {
    if (_dim == std::numeric_limits<uint32_t>::max()) {
      
    }
    if constexpr (std::is_same<T, uint32_t>::value ||
                  std::is_same<T, float>::value) {
      return {{_dim, std::is_same<T, float>::value}};
    }
    return std::nullopt;
  }

  uint64_t numRows() const final { return _data.size(); }

  /**
   * The extra typename keyword here so that during parsing it is clear that
   * ArrayColumn<T>::RowReference refers to a type and not a static member (or
   * something else) within the class.
   * https://stackoverflow.com/questions/60277129/why-is-typename-necessary-in-return-type-c
   * https://en.cppreference.com/w/cpp/language/qualified_lookup
   */
  typename ArrayColumn<T>::RowReference operator[](uint64_t n) const final {
    uint64_t len = _data[n].size();
    const T* ptr = _data[n].data();

    return {ptr, len};
  }

 private:
  static void checkArrayIsValid(
      const std::vector<std::vector<uint32_t>>& data) {
    if (data.empty() || data[0].empty()) {
      throw std::invalid_argument(
          "Can only construct VectorArrayColumn non-empty data.");
    }
  }

  std::vector<std::vector<T>> _data;
  uint32_t _dim;
};

}  // namespace thirdai::dataset