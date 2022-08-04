#pragma once

#include "BoltVector.h"
#include <bolt/src/layers/LayerConfig.h>

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
                                     bool use_sparsity) const = 0;

  virtual void freezeHashTables(bool insert_labels_if_not_found) = 0;

  virtual void buildHashTables() = 0;

  virtual void reBuildHashFunction() = 0;

  // Returns the dimenion of the layer.
  virtual uint32_t getDim() const = 0;

  // Returns the input dimension of the layer
  virtual uint32_t getInputDim() const = 0;

  // Returns the current output dimension for inference (different if sparse
  // inference).
  virtual uint32_t getSparseDim() const = 0;

  virtual float* getWeights() const = 0;

  virtual float* getBiases() const = 0;

  virtual void setWeightGradients(const float* update_weight_gradient) = 0;

  virtual void setBiasesGradients(const float* update_bias_gradient) = 0;

  virtual float* getBiasesGradient() = 0;

  virtual float* getWeightsGradient() = 0;

  virtual void setTrainable(bool trainable) = 0;

  virtual void setWeights(const float* new_weights) = 0;

  virtual bool getTrainable() const = 0;

  virtual void setBiases(const float* new_biases) = 0;

  virtual float getSparsity() const = 0;

  virtual void setSparsity(float sparsity) = 0;

  virtual void buildLayerSummary(std::stringstream& summary,
                                 bool detailed) const {
    (void)detailed;
    summary << "dim=" << getDim() << "\n";
  }

  virtual void getBiasGradientSketch(uint32_t* indices, float* gradients,
                                     uint32_t sketch_size,
                                     int seed_for_hashing) const = 0;

  virtual void getWeightGradientSketch(uint32_t* indices, float* gradients,
                                       uint32_t sketch_size,
                                       int seed_for_hashing) const = 0;

  virtual void setBiasGradientsFromIndicesValues(uint32_t* indices_raw_data,
                                                 float* values_raw_data,
                                                 uint32_t sketch_size) = 0;

  virtual void setWeightGradientsFromIndicesValues(uint32_t* indices_raw_data,
                                                   float* values_raw_data,
                                                   uint32_t sketch_size) = 0;

  virtual void getUnbiasedBiasGradientSketch(int* indices, int sketch_size,
                                             int seed_for_hashing,
                                             bool pregenerate_distribution,
                                             float threshold) const = 0;

  virtual void getUnbiasedWeightGradientSketch(int* indices, int sketch_size,
                                               int seed_for_hashing,
                                               bool pregenerate_distribution,
                                               float threshold) const = 0;

  virtual void setUnbiasedBiasGradientsFromIndicesValues(int* indices_raw_data,
                                                         int sketch_size,
                                                         float threshold) = 0;

  virtual void setUnbiasedWeightGradientsFromIndicesValues(
      int* indices_raw_data, int sketch_size, float threshold) = 0;

  virtual float getUnbiasedBiasThresholdForGradient(int sketch_size) const = 0;

  virtual float getUnbiasedWeightThresholdForGradient(
      int sketch_size) const = 0;

  virtual ~SequentialLayer() = default;
};
}  // namespace thirdai::bolt