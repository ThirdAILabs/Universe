#include "Init.h"
#include <bolt/src/utils/Timer.h>
#include <utils/Logging.h>
#include <algorithm>
#include <random>

namespace thirdai::smx {

DenseTensorPtr zeros(const Shape& shape) { return fill(shape, 0.F); }

DenseTensorPtr ones(const Shape& shape) { return fill(shape, 1.F); }

DenseTensorPtr fill(const Shape& shape, float value) {
  bolt::utils::Timer alloc_timer;
  auto tensor = DenseTensor::make(shape, Dtype::f32);
  alloc_timer.stop();
  std::cerr << "fill alloc shape=" << shape.toString() << " time "
            << alloc_timer.milliseconds() << " ms" << std::endl;

  bolt::utils::Timer fill_timer;

  std::fill_n(tensor->data<float>(), tensor->size(), value);

  fill_timer.stop();
  std::cerr << "fill fill shape=" << shape.toString() << " time "
            << fill_timer.milliseconds() << " ms" << std::endl;
  return tensor;
}

DenseTensorPtr normal(const std::vector<size_t>& shape, float mean,
                      float stddev, uint32_t seed) {
  auto tensor = DenseTensor::make(Shape(shape), Dtype::f32);

  std::normal_distribution<float> dist(mean, stddev);
  std::mt19937 rng(seed);

  float* data = tensor->data<float>();
  size_t len = tensor->size();
  for (size_t i = 0; i < len; i++) {
    data[i] = dist(rng);
  }

  return tensor;
}

DenseTensorPtr scalar(float value) {
  auto tensor = DenseTensor::make(Shape(), Dtype::f32,
                                  DefaultMemoryHandle::allocate(sizeof(float)));
  tensor->data<float>()[0] = value;

  return tensor;
}

}  // namespace thirdai::smx