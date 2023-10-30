#pragma once

#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>

namespace thirdai::mach {

using TrainOptions = automl::udt::TrainOptions;

static float autotuneSparsity(uint32_t dim) {
  return automl::udt::utils::autotuneSparsity(dim);
}

}  // namespace thirdai::mach