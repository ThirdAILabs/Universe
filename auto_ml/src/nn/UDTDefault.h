#pragma once

#include <bolt/src/graph/Graph.h>
#include <cstdint>
#include <vector>

namespace thirdai::automl::nn {

// Constructs a default bolt graph for UDT.
bolt::BoltGraphPtr UDTDefault(const std::vector<uint32_t>& input_dims,
                              uint32_t output_dim, uint32_t hidden_layer_size);

// Constructs a bolt graph for from a saved model config.
bolt::BoltGraphPtr fromConfig(const std::vector<uint32_t>& input_dims,
                              uint32_t output_dim,
                              const std::string& saved_model_config);

}  // namespace thirdai::automl::nn