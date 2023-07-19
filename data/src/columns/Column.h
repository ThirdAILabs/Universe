#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::data {

struct ColumnDimension {
  size_t dim;
  bool is_dense;

  static ColumnDimension dense(size_t dim) {
    return ColumnDimension(dim, /* is_dense= */ true);
  }

  static std::optional<ColumnDimension> sparse(std::optional<size_t> dim) {
    if (!dim) {
      return std::nullopt;
    }
    return ColumnDimension(*dim, /* is_dense= */ false);
  }

  friend bool operator==(const ColumnDimension& a, const ColumnDimension& b) {
    return a.dim == b.dim && a.is_dense == b.is_dense;
  }

  friend bool operator!=(const ColumnDimension& a, const ColumnDimension& b) {
    return !(a == b);
  }

 private:
  ColumnDimension(size_t dim, size_t is_dense) : dim(dim), is_dense(is_dense) {}
};

class Column;
using ColumnPtr = std::shared_ptr<Column>;

class Column {
 public:
  virtual size_t numRows() const = 0;

  virtual std::optional<ColumnDimension> dimension() const = 0;

  virtual void shuffle(const std::vector<size_t>& permutation) = 0;

  virtual ColumnPtr concat(ColumnPtr&& other) = 0;

  virtual std::pair<ColumnPtr, ColumnPtr> split(size_t offset) = 0;

  virtual ~Column() = default;
};

using ColumnPtr = std::shared_ptr<Column>;

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
};

template <typename T>
using ArrayColumnBasePtr = std::shared_ptr<ArrayColumnBase<T>>;

template <typename T>
class ValueColumnBase : public ArrayColumnBase<T> {
 public:
  virtual const T& value(size_t row) const = 0;

  virtual ~ValueColumnBase() = default;
};

template <typename T>
using ValueColumnBasePtr = std::shared_ptr<ValueColumnBase<T>>;

}  // namespace thirdai::data