#pragma once

#include "Tensor.h"
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/InputTensor.h>

namespace thirdai::bolt::nn::tensor {

/**
 * Subclass of Tensor which represents a computation in the model/computation
 * graph as well as the outputs of the computation. It stores the op that
 * generates the output as well as references the input tensor(s) for the
 * computation. This class comprises the core of the computation graph in the
 * model by storing the relationship between inputs and outputs within the
 * model and how information is passed between them.
 */
class ActivationTensor final : public Tensor {
 public:
  ActivationTensor(uint32_t dim, ops::OpPtr source, TensorList inputs);

  static std::shared_ptr<ActivationTensor> make(uint32_t dim, ops::OpPtr source,
                                                TensorList inputs);

  /**
   * Returns the op which whose activations are stored in the tensor.
   */
  ops::OpPtr source() const;

  /**
   * Returns the inputs to the tensor.
   */
  const TensorList& inputs() const;

  /**
   * Computes the activations of the neurons in the tensor from its inputs using
   * its source op. Calls the forward method of the source op.
   */
  void forward(uint32_t index_in_batch, bool training);

  /**
   * Backpropagates the gradients of the tensor to its inputs using the source
   * op. Calls the backpropagate method of the source op.
   */
  void backpropagate(uint32_t index_in_batch);

  /**
   * Returns the number of nonzeros the tensor will contain depending on wether
   * or not sparsity is being used and the inputs. Calls the numNonzeros method
   * of the source op.
   */
  std::optional<uint32_t> numNonzeros(bool use_sparsity) const final;

  /**
   * Returns the ith vector stored in the tensor. This will be the ith output of
   * the op the last time it was computed.
   */
  BoltVector& getVector(uint32_t index) final;

  /**
   * Reallocates the number of vectors stored in the Tensor to reflect either a
   * change in the batch size the model is processing, a change in whether
   * sparsity is being used for the computations, or a change in the sparsity of
   * some op in the model. This method obtains its number of nonzeros from its
   * source op by passing in the inputs and wether sparsity is enabled.
   */
  void allocate(uint32_t batch_size, bool use_sparsity);

  /**
   * Adds an additional input to the tensor which will be passed into its source
   * op during forward and backward. This is only intended to be used to add the
   * labels as an extra input to the FullyConnected op when the FullyConnected
   * op yields an output and we want the labels to always be in the set of
   * active neurons.
   */
  void addInput(InputTensorPtr input);

  /**
   * Returns the current shape of the active neurons, activations, and
   * gradients.
   */
  std::vector<uint32_t> shape() const;

  /**
   * Returns a non-owning ptr to the active neurons.
   */
  const uint32_t* activeNeuronsPtr() const;

  /**
   * Returns a non-owning ptr to the activations.
   */
  const float* activationsPtr() const;

  /**
   * Returns a non-owning ptr to the gradients.
   */
  const float* gradientsPtr() const;

 private:
  ops::OpPtr _source;
  TensorList _inputs;

  std::vector<BoltVector> _vectors;

  // Storing the activations and active neurons as a continuous array and taking
  // pointers into it allows us to map the activations to numpy arrays without
  // copying.
  std::vector<uint32_t> _active_neurons;
  std::vector<float> _activations;
  std::vector<float> _gradients;
};

using ActivationTensorPtr = std::shared_ptr<ActivationTensor>;

ActivationTensorPtr asActivationTensor(const tensor::TensorPtr& tensor);

}  // namespace thirdai::bolt::nn::tensor