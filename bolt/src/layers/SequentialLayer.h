#pragma once

#include "BoltVector.h"
#include <iostream>

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


  virtual void setWeightGradients(const float* update_weight_gradient) = 0;
  
  virtual void setBiasesGradients(const float* update_bias_gradient) = 0;

  virtual float* getBiasesGradient() = 0;
  
  virtual float* getWeightsGradient() = 0;

  virtual void setTrainable(bool trainable) = 0;

  virtual void setWeights(const float* new_weights) = 0;

  virtual bool getTrainable() = 0;

  virtual void setBiases(const float* new_biases) = 0;

  /**
   * Sets whether the layer is currently shallow (shallow
   * means that it has the minimum amount of parameters
   * necessary for inference). This can involve initializing or
   * deleting optimizer state.
   */
  virtual void setShallow(bool shallow) = 0;

  /**
   * Checks whether the layer is shallow, ie, it's optimizer is initialized or
   * uninitialized.
   */
  virtual bool isShallow() = 0;

  /**
   * Sets the save parameter for a layer indicating whether the layer should be
   * saved with or without the optimizer state.
   */
  virtual void setShallowSave(bool shallow) = 0;
  virtual void buildLayerSummary(std::stringstream& summary, bool detailed) {
    (void)detailed;
    summary << "dim=" << getDim() << "\n";
  }

  virtual ~SequentialLayer() = default;
};
}  // namespace thirdai::bolt