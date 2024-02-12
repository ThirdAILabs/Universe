#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::data {

class Column;
using ColumnPtr = std::shared_ptr<Column>;

class Column {
 public:
  virtual size_t numRows() const = 0;

  /**
   * Returns the dimension of the column if the column has a known dimension.
   * For some columns it may not be possible to assign a dimension, in which
   * case this will also return nullopt. For example string columns, decimal
   * columns with varying row sizes, or token columns without a max token value
   * will not have a dimension.
   */
  virtual std::optional<size_t> dim() const = 0;

  /**
   * Applies the permutation to the column in place.
   */
  virtual void shuffle(const std::vector<size_t>& permutation) = 0;

  /**
   * Creates a new column with the given permutation.
   */
  virtual ColumnPtr permute(const std::vector<size_t>& permutation) const = 0;

  /**
   * Concatenates the column with another column and returns a new column. Moves
   * the values out of both of the original columns to avoid expensive copies.
   */
  virtual ColumnPtr concat(ColumnPtr&& other) = 0;

  /**
   * Splits the column into two columns. The first returned column will have
   * elements [0, starting_offset) and the second column will have elements
   * [starting_offset, num_rows). This will consume the current column and move
   * its values to avoid copies.
   */
  virtual std::pair<ColumnPtr, ColumnPtr> split(size_t starting_offset) = 0;

  virtual ~Column() = default;
};

template <typename T>
class RowView {
 public:
  RowView(const T* data, size_t len) : _data(data), _len(len) {}

  const T& operator[](size_t i) const {
    if (i >= _len) {
      throw std::out_of_range("Cannot access element " + std::to_string(i) +
                              " of Rowreference of length " +
                              std::to_string(_len) + ".");
    }
    return _data[i];
  }

  std::vector<T> range(size_t start, size_t end) const {
    if (end < start || _len < end) {
      throw std::out_of_range("Invalid range [" + std::to_string(start) + ", " +
                              std::to_string(end) + ") for row with length " +
                              std::to_string(_len) + ".");
    }

    return {begin() + start, begin() + end};
  }

  std::vector<T> toVector() const { return {begin(), end()}; }

  size_t size() const { return _len; }

  const T* data() const { return _data; }

  const T* begin() const { return _data; }

  const T* end() const { return _data + _len; }

 private:
  const T* _data;
  size_t _len;
};

template <typename T>
class ArrayColumnBase : public Column {
 public:
  virtual RowView<T> row(size_t row) const = 0;

  virtual ~ArrayColumnBase() = default;

  static auto cast(const ColumnPtr& column) {
    return std::dynamic_pointer_cast<ArrayColumnBase<T>>(column);
  }
};

template <typename T>
using ArrayColumnBasePtr = std::shared_ptr<ArrayColumnBase<T>>;

template <typename T>
class ValueColumnBase : public ArrayColumnBase<T> {
 public:
  virtual const T& value(size_t row) const = 0;

  virtual ~ValueColumnBase() = default;

  static auto cast(const ColumnPtr& column) {
    return std::dynamic_pointer_cast<ValueColumnBase<T>>(column);
  }
};

template <typename T>
using ValueColumnBasePtr = std::shared_ptr<ValueColumnBase<T>>;

}  // namespace thirdai::data