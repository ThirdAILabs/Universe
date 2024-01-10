#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::smx {

class Shape {
 public:
  explicit Shape(std::vector<size_t> shape) : _shape(std::move(shape)) {}

  template <typename... Dims>
  explicit Shape(Dims... shape) : Shape(std::vector<size_t>{shape...}) {}

  size_t ndim() const { return _shape.size(); }

  size_t size() const {
    return std::reduce(_shape.begin(), _shape.end(), 1, std::multiplies<>{});
  }

  bool isScalar() const { return _shape.empty(); }

  auto begin() const { return _shape.begin(); }

  auto end() const { return _shape.end(); }

  Shape slice(size_t start, size_t end) const {
    if (start > end || end > ndim()) {
      throw std::invalid_argument(
          "Cannot slice shape with ndim=" + std::to_string(ndim()) +
          " with indices (" + std::to_string(start) + ", " +
          std::to_string(end) + ").");
    }
    return Shape(std::vector<size_t>{begin() + start, begin() + end});
  }

  Shape slice(size_t start) const { return slice(start, ndim()); }

  size_t operator[](size_t dim) const {
    if (dim >= _shape.size()) {
      throw std::out_of_range("Cannot access dim " + std::to_string(dim) +
                              " of shape with ndim=" + std::to_string(ndim()) +
                              ".");
    }
    return _shape[dim];
  }

  size_t last() const { return _shape.back(); }

  const auto& vector() const { return _shape; }

  bool canReshapeTo(const Shape& other) const { return size() == other.size(); }

  Shape permute(const std::vector<size_t>& perm) const {
    if (perm.size() != ndim()) {
      throw std::invalid_argument(
          "All dimensions must be specified in permutation.");
    }

    std::vector<size_t> perm_shape;
    perm_shape.reserve(ndim());

    std::vector<bool> dims_used(ndim(), false);
    for (size_t p : perm) {
      if (p >= ndim()) {
        throw std::out_of_range("Invalid index " + std::to_string(p) +
                                " for permutation of shape with " +
                                std::to_string(ndim()) + " dimensions.");
      }
      perm_shape.push_back(_shape[p]);
      dims_used[p] = true;
    }

    if (!std::all_of(dims_used.begin(), dims_used.end(),
                     [](bool x) { return x; })) {
      throw std::invalid_argument(
          "All dimensions must be specified in permutation.");
    }

    return Shape(std::move(perm_shape));
  }

  std::string toString() const {
    std::stringstream str;
    str << "(";
    for (size_t i = 0; i < _shape.size(); i++) {
      if (i > 0) {
        str << ", ";
      }
      str << _shape[i];
    }
    str << ")";
    return str.str();
  }

  friend bool operator==(const Shape& a, const Shape& b) {
    return b._shape == a._shape;
  }

  friend bool operator!=(const Shape& a, const Shape& b) { return !(a == b); }

 private:
  std::vector<size_t> _shape;
};

}  // namespace thirdai::smx