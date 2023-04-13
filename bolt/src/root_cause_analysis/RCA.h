#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::nn::rca {

using RCAInputGradients =
    std::pair<std::optional<std::vector<uint32_t>>, std::vector<float>>;

RCAInputGradients explainPrediction(model::ModelPtr& model,
                                    const BoltVector& input_vec);

RCAInputGradients explainNeuron(model::ModelPtr& model,
                                const BoltVector& input_vec, uint32_t neuron);

}  // namespace thirdai::bolt::nn::rca