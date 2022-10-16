#pragma once

#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

class Column {
 public:
  virtual uint64_t numRows() const = 0;

  virtual bool isDense() const = 0;

  virtual void appendRowToVector(SegmentedFeatureVector& vector,
                                 uint64_t row) const = 0;

  virtual ~Column() = default;
};

using ColumnPtr = std::shared_ptr<Column>;

template <typename T>
class ValueColumn : public Column {
 public:
  virtual const T& operator[](size_t n) const = 0;

  bool isDense() const final { return std::is_same<T, float>::value; }

  void appendRowToVector(SegmentedFeatureVector& vector,
                         uint64_t row) const final {
    if constexpr (std::is_same<T, uint32_t>::value) {
      vector.addSparseFeatureToSegment(this->operator[](row), 1.0);
      return;
    }

    if constexpr (std::is_same<T, float>::value) {
      vector.addDenseFeatureToSegment(this->operator[](row));
      return;
    }

    throw std::runtime_error(
        "Cannot convert ValueColumn to BoltVector if its type is not int or "
        "float.");
  }

  virtual ~ValueColumn() = default;
};

using IntegerValueColumn = ValueColumn<uint32_t>;
using FloatValueColumn = ValueColumn<float>;

template <typename T>
class ArrayColumn : public Column {
 public:
  class RowReference {
   public:
    RowReference(const T* data, uint64_t len) : _data(data), _len(len) {}

    const T& operator[](uint64_t i) const {
      if (i >= _len) {
        throw std::out_of_range("Cannot access element " + std::to_string(i) +
                                " of Rowreference of length " +
                                std::to_string(_len) + ".");
      }
      return *(_data + i);
    }

    uint64_t size() const { return _len; }

    const T* begin() const { return _data; }

    const T* end() const { return _data + _len; }

   private:
    const T* _data;
    uint64_t _len;
  };

  virtual RowReference operator[](size_t n) const = 0;

  bool isDense() const final { return std::is_same<T, float>::value; }

  void appendRowToVector(SegmentedFeatureVector& vector,
                         uint64_t row) const final {
    if constexpr (std::is_same<T, uint32_t>::value) {
      for (uint32_t index : this->operator[](row)) {
        vector.addSparseFeatureToSegment(index, 1.0);
      }
      return;
    }

    if constexpr (std::is_same<T, float>::value) {
      for (uint32_t value : this->operator[](row)) {
        vector.addDenseFeatureToSegment(value);
      }
      return;
    }

    if constexpr (std::is_same<T, std::pair<uint32_t, float>>::value) {
      for (const auto& [index, value] : this->operator[](row)) {
        vector.addSparseFeatureToSegment(index, value);
      }
      return;
    }

    throw std::runtime_error(
        "Cannot convert ArrayColumn to BoltVector if its type is not int, "
        "float, or (int, float) pair.");
  }

  virtual ~ArrayColumn() = default;
};

using IntegerArrayColumn = ArrayColumn<uint32_t>;
using FloatArrayColumn = ArrayColumn<float>;

}  // namespace thirdai::dataset