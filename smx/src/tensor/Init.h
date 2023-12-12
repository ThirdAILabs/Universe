#pragma once

#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Tensor.h>
#include <utils/Random.h>

namespace thirdai::smx {

DenseTensorPtr zeros(const std::vector<size_t>& shape);

DenseTensorPtr ones(const std::vector<size_t>& shape);

DenseTensorPtr fill(const std::vector<size_t>& shape, float value);

DenseTensorPtr normal(const std::vector<size_t>& shape, float mean,
                      float stddev, uint32_t seed = global_random::nextSeed());

}  // namespace thirdai::smx