#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/config/ArgumentMap.h>

namespace thirdai::automl::udt::utils {

bolt::BoltGraphPtr buildModel(uint32_t input_dim, uint32_t output_dim,
                              const config::ArgumentMap& args,
                              const std::optional<std::string>& model_config,
                              bool use_sigmoid_bce = false);

bolt::BoltGraphPtr defaultModel(uint32_t input_dim, uint32_t hidden_dim,
                                uint32_t output_dim,
                                bool use_sigmoid_bce = false);

bolt::BoltGraphPtr loadModel(const std::vector<uint32_t>& input_dims,
                             uint32_t output_dim,
                             const std::string& config_path);

bool hasSoftmaxOutput(const bolt::BoltGraphPtr& model);

}  // namespace thirdai::automl::udt::utils