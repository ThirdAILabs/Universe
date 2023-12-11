#pragma once

#include <smx/src/tensor/Dtype.h>
#include <smx/src/tensor/Shape.h>
#include <memory>
#include <stdexcept>

namespace thirdai::smx {

class Tensor {
 public:
  Tensor(Shape shape, Dtype dtype) : _shape(std::move(shape)), _dtype(dtype) {}

  const Shape& shape() const { return _shape; }

  size_t shapeAt(size_t dim) const { return _shape[dim]; }

  size_t ndim() const { return _shape.ndim(); }

  Dtype dtype() const { return _dtype; }

  virtual bool isSparse() const = 0;

  virtual ~Tensor() = default;

 protected:
  Shape _shape;
  Dtype _dtype;
};

using TensorPtr = std::shared_ptr<Tensor>;

// NOLINTNEXTLINE
#define CHECK(stmt, msg)                                                   \
  if (!(stmt)) {                                                           \
    throw std::runtime_error(std::string(__FILE__) + ":" +                 \
                             std::string(std::to_string(__LINE__)) + " " + \
                             (msg));                                       \
  }

}  // namespace thirdai::smx