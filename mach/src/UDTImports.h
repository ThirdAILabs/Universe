#pragma once

#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>

namespace thirdai::mach {

using TrainOptions = automl::udt::TrainOptions;

using ArgumentMap = automl::config::ArgumentMap;

static float autotuneSparsity(uint32_t dim) {
  return automl::udt::utils::autotuneSparsity(dim);
}

static auto buildModel(uint32_t input_dim, uint32_t num_buckets,
                       const automl::config::ArgumentMap& args,
                       const std::optional<std::string>& model_config,
                       bool use_sigmoid_bce) {
  return automl::udt::utils::buildModel(input_dim, num_buckets, args,
                                        model_config, use_sigmoid_bce,
                                        /* mach= */ true);
}

}  // namespace thirdai::mach