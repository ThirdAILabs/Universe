#pragma once

#include "BoltVector.h"

namespace thirdai::bolt {
class SequentialLayer {
 public:
  virtual void forward(const BoltVector& input, BoltVector& output,
                       const BoltVector* labels) = 0;

  virtual void backpropagate(BoltVector& input, BoltVector& output) = 0;

  virtual void backpropagateInputLayer(BoltVector& input,
                                       BoltVector& output) = 0;

  virtual void updateParameters(float lr, uint32_t iter, float B1, float B2,
                                float eps) = 0;

  virtual BoltBatch createBatchState(uint32_t batch_size,
                                     bool force_dense) const = 0;

  virtual void forceSparseForInference() = 0;

  virtual bool isForceSparsity() const = 0;

  virtual void buildHashTables() = 0;

  virtual void reBuildHashFunction() = 0;

  virtual void shuffleRandNeurons() = 0;

  // Returns the dimenion of the layer.
  virtual uint32_t getDim() const = 0;

  // Returns the input dimension of the layer
  virtual uint32_t getInputDim() const = 0;

  // Returns the current output dimension for inference (different if sparse
  // inference).
  virtual uint32_t getInferenceOutputDim() const = 0;

  virtual float* getWeights() = 0;

  virtual float* getBiases() = 0;

  virtual void setWeights(const float* new_weights) = 0;

  virtual void setBiases(const float* new_biases) = 0;

  virtual float* getWeightGradients() = 0;
  virtual float* getBiasGradients() = 0;
  virtual float* getWeightMomentum() = 0;
  virtual float* getBiasMomentum() = 0;
  virtual float* getWeightVelocity() = 0;
  virtual float* getBiasVelocity() = 0;

  virtual void setWeightGradients(const float* new_weight_gradients) = 0;
  virtual void setBiasGradients(const float* new_bias_gradients) = 0;
  virtual void setWeightMomentum(const float* new_weight_momentum) = 0;
  virtual void setBiasMomentum(const float* new_bias_momentum) = 0;
  virtual void setWeightVelocity(const float* new_weight_velocity) = 0;
  virtual void setBiasVelocity(const float* new_bias_velocity) = 0;

  virtual ~SequentialLayer() = default;
};
}  // namespace thirdai::bolt