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

  ColumnDimension(size_t dim, size_t is_dense) : dim(dim), is_dense(is_dense) {}

  friend bool operator==(const ColumnDimension& a, const ColumnDimension& b) {
    return a.dim == b.dim && a.is_dense == b.is_dense;
  }

  friend bool operator!=(const ColumnDimension& a, const ColumnDimension& b) {
    return !(a == b);
  }
};

class Column {
 public:
  virtual size_t numRows() const = 0;

  virtual std::optional<ColumnDimension> dimension() const = 0;

  virtual void shuffle(const std::vector<size_t>& permutation) = 0;

  virtual std::shared_ptr<Column> concat(std::shared_ptr<Column>&& other) = 0;

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
class ArrayColumn : public Column {
 public:
  virtual RowView<T> row(size_t row) const = 0;

  virtual ~ArrayColumn() = default;
};

template <typename T>
using ArrayColumnPtr = std::shared_ptr<ArrayColumn<T>>;

template <typename T>
class ValueColumn : public ArrayColumn<T> {
 public:
  virtual const T& value(size_t row) const = 0;

  virtual ~ValueColumn() = default;
};

template <typename T>
using ValueColumnPtr = std::shared_ptr<ValueColumn<T>>;

}  // namespace thirdai::data