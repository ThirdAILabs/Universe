#pragma once

#include <bolt/src/graph/Graph.h>

namespace thirdai::automl::udt::utils {

bolt::BoltGraphPtr defaultModel(uint32_t input_dim, uint32_t hidden_dim,
                                uint32_t output_dim);

}  // namespace thirdai::automl::udt::utils