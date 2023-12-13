#pragma once

#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Tensor.h>
#include <utils/Random.h>

namespace thirdai::smx {

DenseTensorPtr zeros(const Shape& shape);

inline DenseTensorPtr zeros(const std::vector<size_t>& shape) {
  return zeros(Shape(shape));
}

DenseTensorPtr ones(const Shape& shape);

inline DenseTensorPtr ones(const std::vector<size_t>& shape) {
  return ones(Shape(shape));
}

DenseTensorPtr fill(const Shape& shape, float value);

inline DenseTensorPtr fill(const std::vector<size_t>& shape, float value) {
  return fill(Shape(shape), value);
}

DenseTensorPtr normal(const std::vector<size_t>& shape, float mean,
                      float stddev, uint32_t seed = global_random::nextSeed());

DenseTensorPtr scalar(float value);

}  // namespace thirdai::smx