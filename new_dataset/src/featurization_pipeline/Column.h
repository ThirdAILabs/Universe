#pragma once

#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace thirdai::data::columns {

struct DimensionInfo {
  uint32_t dim;
  bool is_dense;
};

template <typename T>
struct Contribution {
  Contribution() {}
  Contribution(T value, float gradient) : value(value), gradient(gradient) {}
  T value;
  float gradient;
  std::string time_frame;
};

class Column {
 public:
  virtual uint64_t numRows() const = 0;

  /**
   * This method returns information about the dimension of the column. This is
   * used to concatenate columns into a bolt vector. Returning nullopt means
   * that the column cannot be concatenated, this would be used things like
   * string columns.
   */
  virtual std::optional<DimensionInfo> dimension() const = 0;

  virtual void appendRowToVector(dataset::SegmentedFeatureVector& vector,
                                 uint64_t row_idx) const = 0;

  virtual ~Column() = default;
};

using ColumnPtr = std::shared_ptr<Column>;

// We use templates to create columns with different types because there are
// very few types which we will need to support and almost all of the code for
// the columns of different types is the same.
template <typename T>
class ValueColumn : public Column {
 public:
  virtual const T& operator[](uint64_t n) const = 0;

  void appendRowToVector(dataset::SegmentedFeatureVector& vector,
                         uint64_t row_idx) const final {
    if constexpr (std::is_same<T, uint32_t>::value) {
      vector.addSparseFeatureToSegment(this->operator[](row_idx), 1.0);
      return;
    }

    if constexpr (std::is_same<T, float>::value) {
      vector.addDenseFeatureToSegment(this->operator[](row_idx));
      return;
    }

    if constexpr (std::is_same<T, std::pair<uint32_t, float>>::value) {
      std::pair<uint32_t, float> index_value = this->operator[](row_idx);
      vector.addSparseFeatureToSegment(index_value.first, index_value.second);
      return;
    }

    // Need this void here because if none of the above constexprs match, then
    // row_idx will be unused
    (void)row_idx;

    throw std::runtime_error(
        "Cannot convert ValueColumn to BoltVector if its type is not int or "
        "float.");
  }

  virtual ~ValueColumn() = default;
};

using TokenColumn = ValueColumn<uint32_t>;
using DenseFeatureColumn = ValueColumn<float>;
using SparseFeatureColumn = ValueColumn<std::pair<uint32_t, float>>;
using StringColumn = ValueColumn<std::string>;

using TokenColumnPtr = std::shared_ptr<TokenColumn>;
using DenseFeatureColumnPtr = std::shared_ptr<DenseFeatureColumn>;
using SparseFeatureColumnPtr = std::shared_ptr<SparseFeatureColumn>;
using StringColumnPtr = std::shared_ptr<StringColumn>;

// We use templates to create columns with different types because there are
// very few types which we will need to support and almost all of the code for
// the columns of different types is the same.
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
      return _data[i];
    }

    uint64_t size() const { return _len; }

    const T* begin() const { return _data; }

    const T* end() const { return _data + _len; }

   private:
    const T* _data;
    uint64_t _len;
  };

  virtual RowReference operator[](uint64_t n) const = 0;

  void appendRowToVector(dataset::SegmentedFeatureVector& vector,
                         uint64_t row_idx) const final {
    static_assert(std::is_same<T, uint32_t>::value ||
                      std::is_same<T, float>::value ||
                      std::is_same<T, std::pair<uint32_t, float>>::value,
                  "Can only convert columns of type uint32, float32, or "
                  "(uint32, float32) to BoltVector.");

    if constexpr (std::is_same<T, uint32_t>::value) {
      for (uint32_t index : this->operator[](row_idx)) {
        vector.addSparseFeatureToSegment(index, 1.0);
      }
    }

    if constexpr (std::is_same<T, float>::value) {
      for (float value : this->operator[](row_idx)) {
        vector.addDenseFeatureToSegment(value);
      }
    }

    if constexpr (std::is_same<T, std::pair<uint32_t, float>>::value) {
      for (const auto& [index, value] : this->operator[](row_idx)) {
        vector.addSparseFeatureToSegment(index, value);
      }
    }
  }

  virtual ~ArrayColumn() = default;
};

using TokenArrayColumn = ArrayColumn<uint32_t>;
using DenseArrayColumn = ArrayColumn<float>;
using SparseArrayColumn = ArrayColumn<std::pair<uint32_t, float>>;

using TokenArrayColumnPtr = std::shared_ptr<TokenArrayColumn>;
using DenseArrayColumnPtr = std::shared_ptr<DenseArrayColumn>;
using SparseArrayColumnPtr = std::shared_ptr<SparseArrayColumn>;

class ContibutionColumnBase {
 public:
  virtual uint64_t numRows() const = 0;

  virtual ~ContibutionColumnBase() = default;
};

using ContibutionColumnBasePtr = std::shared_ptr<ContibutionColumnBase>;

template <typename T>
class ContibutionColumn : public ContibutionColumnBase {
 public:
  virtual std::vector<Contribution<T>> getRow(uint64_t n) const = 0;

  virtual void resize(uint64_t n) = 0;

  virtual void insert(const std::vector<Contribution<T>>& row_values,
                      uint64_t n) = 0;

  virtual ~ContibutionColumn() = default;
};

using TokenContributionColumn = ContibutionColumn<uint32_t>;
using StringContributionColumn = ContibutionColumn<std::string>;

using TokenContributionColumnPtr = std::shared_ptr<TokenContributionColumn>;

}  // namespace thirdai::data::columns