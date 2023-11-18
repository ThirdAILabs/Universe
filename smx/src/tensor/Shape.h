#pragma once

#include <functional>
#include <numeric>
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

  auto begin() const { return _shape.begin(); }

  auto end() const { return _shape.end(); }

  size_t operator[](size_t dim) const {
    if (dim >= _shape.size()) {
      throw std::out_of_range("Cannot access dim " + std::to_string(dim) +
                              " of shape with ndim=" + std::to_string(ndim()) +
                              ".");
    }
    return _shape[dim];
  }

  const auto& vector() const { return _shape; }

 private:
  std::vector<size_t> _shape;
};

}  // namespace thirdai::smx