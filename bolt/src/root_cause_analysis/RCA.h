#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::nn::rca {

/**
 * Stores the input gradients returned by RCA explainability methods. The
 * indices are only used if the input is sparse.
 */
struct RCAGradients {
  std::optional<std::vector<uint32_t>> indices;
  std::vector<float> gradients;
};

/**
 * Commutes the gradients w.r.t the inputs by assigning a label of 1.0 to the
 * correct label and computing gradients. These input gradients then represent
 * what changes in the input would make the activation of the predicted neuron
 * closer to 1.0.
 */
RCAGradients explainPrediction(model::ModelPtr& model,
                               const tensor::TensorList& input_vec);

/**
 * Commutes the gradients w.r.t the inputs by assigning a label of 1.0 to the
 * specified neuron and computing gradients. These input gradients then
 * represent what changes in the input would make the activation of the given
 * neuron closer to 1.0.
 */
RCAGradients explainNeuron(model::ModelPtr& model,
                           const tensor::TensorList& input_vec,
                           uint32_t neuron);

}  // namespace thirdai::bolt::nn::rca