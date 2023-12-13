#include "Init.h"
#include <algorithm>
#include <random>

namespace thirdai::smx {

DenseTensorPtr zeros(const Shape& shape) { return fill(shape, 0.F); }

DenseTensorPtr ones(const Shape& shape) { return fill(shape, 1.F); }

DenseTensorPtr fill(const Shape& shape, float value) {
  auto tensor = DenseTensor::make(shape, Dtype::f32);
  std::fill_n(tensor->data<float>(), tensor->shape().size(), value);
  return tensor;
}

DenseTensorPtr normal(const std::vector<size_t>& shape, float mean,
                      float stddev, uint32_t seed) {
  auto tensor = DenseTensor::make(Shape(shape), Dtype::f32);

  std::normal_distribution<float> dist(mean, stddev);
  std::mt19937 rng(seed);

  float* data = tensor->data<float>();
  size_t len = tensor->shape().size();
  for (size_t i = 0; i < len; i++) {
    data[i] = dist(rng);
  }

  return tensor;
}

DenseTensorPtr scalar(float value) {
  auto tensor = DenseTensor::make(Shape(1UL), Dtype::f32);
  tensor->data<float>()[0] = value;

  return tensor;
}

}  // namespace thirdai::smx