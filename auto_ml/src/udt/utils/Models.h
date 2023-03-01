#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/Graph.h>

namespace thirdai::automl::udt::utils {

bolt::BoltGraphPtr defaultModel(uint32_t input_dim, uint32_t hidden_dim,
                                uint32_t output_dim);

bolt::BoltGraphPtr loadModel(const std::vector<uint32_t>& input_dims,
                             uint32_t output_dim,
                             const std::string& config_path);

bool hasSoftmaxOutput(const bolt::BoltGraphPtr& model);

void setModel(bolt::BoltGraphPtr& current_model, bolt::BoltGraphPtr& new_model);

}  // namespace thirdai::automl::udt::utils