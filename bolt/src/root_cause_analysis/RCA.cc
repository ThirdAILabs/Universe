#include "RCA.h"
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <stdexcept>

namespace thirdai::bolt::nn::rca {

tensor::TensorPtr createTensorWithGrad(const tensor::TensorList& inputs,
                                       uint32_t dim) {
  if (inputs.size() != 1 || inputs.at(0)->batchSize() != 1) {
    throw std::invalid_argument(
        "RCA is only supported for single inputs and batch size 1.");
  }

  const BoltVector& vector = inputs.at(0)->getVector(0);

  tensor::TensorPtr tensor;
  if (!vector.isDense()) {
    tensor = tensor::Tensor::sparse(/* batch_size= */ 1, /* dim= */ dim,
                                    /* nonzeros= */ vector.len);

    for (uint32_t i = 0; i < vector.len; i++) {
      if (vector.active_neurons[i] >= dim) {
        throw std::invalid_argument(
            "Recieved input index " + std::to_string(vector.active_neurons[i]) +
            " for input with dimension " + std::to_string(dim) + ".");
      }
      tensor->getVector(0).active_neurons[i] = vector.active_neurons[i];
    }
  } else {
    tensor = tensor::Tensor::dense(/* batch_size= */ 1, /* dim= */ dim);
  }

  std::copy(vector.activations, vector.activations + vector.len,
            tensor->getVector(0).activations);

  return tensor;
}

RCAGradients explainNeuronHelper(model::ModelPtr& model,
                                 const tensor::TensorPtr& input,
                                 uint32_t neuron) {
  auto label = tensor::Tensor::sparse(/* batch_size= */ 1,
                                      /* dim= */ model->labelDims().at(0),
                                      /* nonzeros= */ 1);
  label->getVector(0).active_neurons[0] = neuron;
  label->getVector(0).activations[0] = 1.0;

  model->trainOnBatch({input}, {label});

  RCAGradients gradients;

  const BoltVector& input_vec = input->getVector(0);
  if (!input_vec.isDense()) {
    gradients.indices = std::vector<uint32_t>(
        input_vec.active_neurons, input_vec.active_neurons + input_vec.len);
  }
  gradients.gradients.assign(input_vec.gradients,
                             input_vec.gradients + input_vec.len);

  return gradients;
}

RCAGradients explainPrediction(model::ModelPtr& model,
                               const tensor::TensorList& input_vec) {
  if (model->inputDims().size() != 1 || model->outputs().size() != 1) {
    throw std::invalid_argument(
        "RCA is only supported for models with a single input and output.");
  }

  auto input = createTensorWithGrad(input_vec, model->inputDims()[0]);

  // TODO(Nicholas): Should we use sparsity?
  auto output = model->forward({input}, /* use_sparsity= */ false).at(0);

  uint32_t prediction = output->getVector(0).getHighestActivationId();

  return explainNeuronHelper(model, input, prediction);
}

RCAGradients explainNeuron(model::ModelPtr& model,
                           const tensor::TensorList& input_vec,
                           uint32_t neuron) {
  if (model->inputDims().size() != 1 || model->outputs().size() != 1) {
    throw std::invalid_argument(
        "RCA is only supported for models with a single input and output.");
  }

  auto input = createTensorWithGrad(input_vec, model->inputDims()[0]);

  return explainNeuronHelper(model, input, neuron);
}

}  // namespace thirdai::bolt::nn::rca