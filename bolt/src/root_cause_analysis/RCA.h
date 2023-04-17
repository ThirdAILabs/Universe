#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::nn::rca {

struct RCAGradients {
  std::optional<std::vector<uint32_t>> indices;
  std::vector<float> gradients;
};

RCAGradients explainPrediction(model::ModelPtr& model,
                               const tensor::TensorList& input_vec);

RCAGradients explainNeuron(model::ModelPtr& model,
                           const tensor::TensorList& input_vec,
                           uint32_t neuron);

}  // namespace thirdai::bolt::nn::rca