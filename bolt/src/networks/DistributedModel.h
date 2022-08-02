#pragma once

#include <wrappers/src/LicenseWrapper.h>
#include <cereal/types/vector.hpp>
#include "Model.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/DatasetLoaders.h>
#include <dataset/src/Datasets.h>
#include <exceptions/src/Exceptions.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace thirdai::bolt {

class DistributedModel : public FullyConnectedNetwork {
 public:
  DistributedModel(SequentialConfigList configs, uint64_t input_dim)
      : FullyConnectedNetwork(std::move(configs), input_dim, true),
        _rebuild_batch(0),
        _rehash_batch(0),
        _train_data(nullptr),
        _train_labels(nullptr) {}

  // Distributed Functions
  uint32_t prepareNodeForDistributedTraining(
      std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>>& train_data,
      const dataset::BoltDatasetPtr& train_labels, uint32_t rehash,
      uint32_t rebuild, bool verbose);

  /*
   * This function calculates the gradient using the train_data of
   * particular batch(provided using batch_no) and for a particular
   * loss_function.
   */
  void calculateGradientSingleNode(uint32_t batch_idx,
                                   const LossFunction& loss_fn);

  /*
   * This function updates the parameters for the neural network
   * using the updated gradients.
   * Right now, the updates are dense meaning that every parameter
   * is getting updated irrespective of type of training(dense or sparse)
   */
  void updateParametersSingleNode(float learning_rate);

  uint32_t numLayers() const;

  uint32_t getDim(uint32_t layer_index) const;

  uint32_t getInputDim() const;

  float* getWeights(uint32_t layer_index);

  float* getBiases(uint32_t layer_index);

  float* getWeightsGradient(uint32_t layer_index);

  float* getBiasesGradient(uint32_t layer_index);

  void setWeights(uint32_t layer_index, const float* data);

  void setBiases(uint32_t layer_index, const float* data);

  void setWeightGradients(uint32_t layer_index, const float* data);

  void setBiasesGradients(uint32_t layer_index, const float* data);

  void getBiasGradientSketch(uint32_t layer_index, uint64_t* indices,
                             float* gradients, uint64_t sketch_size,
                             int seed_for_hashing);

  void getWeightGradientSketch(uint32_t layer_index, uint64_t* indices,
                               float* gradients, uint64_t sketch_size,
                               int seed_for_hashing);

  void setBiasGradientsFromIndicesValues(uint32_t layer_index,
                                         uint64_t* indices_raw_data,
                                         float* values_raw_data,
                                         uint64_t sketch_size);

  void setWeightGradientsFromIndicesValues(uint32_t layer_index,
                                           uint64_t* indices_raw_data,
                                           float* values_raw_data,
                                           uint64_t sketch_size);

 private:
  /*
   outputs here, holds the activations values for output layer,
   between initialization and calculating gradients
  */
  BoltBatch _outputs;

  uint32_t _rebuild_batch;
  uint32_t _rehash_batch;
  std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>> _train_data;
  dataset::BoltDatasetPtr _train_labels;
};

}  // namespace thirdai::bolt
