#pragma once

#include "Model.h"
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <bolt/src/layers/BoltVector.h>
#include <wrappers/src/LicenseWrapper.h>
#include <cereal/types/vector.hpp>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <exceptions/src/Exceptions.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace thirdai::bolt {

class DistributedModel: Model<bolt::BoltBatch>{
 public:
  enum get_type{
    get_weights,
    get_biases,
    get_weights_gradients,
    get_biases_gradients
  };

  enum set_type{
    set_weights,
    set_biases,
    set_weights_gradients,
    set_biases_gradients
  };

  DistributedModel(SequentialConfigList configs, uint64_t input_dim)
      : DistributedNetwork(std::move(configs), input_dim, true),
        _batch_iter(0),
        _epoch_count(0),
        _rebuild_batch(0),
        _rehash_batch(0),
        _train_data(nullptr),
        _train_labels(nullptr){
        thirdai::licensing::LicenseWrapper::checkLicense();
  }

  /**
   * This function takes in a test dataset and uses it to evaluate the model. It
   * returns the final accuracy. The batch_limit parameter limits the number of
   * test batches used, this is intended for intermediate accuracy checks during
   * training with large datasets. Metrics can be passed in to be computed for
   * the test set, and the function optionally store the activations for the
   * output layer in the output_activations array.
   */
  InferenceMetricData predictDistributed(
      // Test dataset
      const std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>>& test_data,
      // Test labels
      const dataset::BoltDatasetPtr& test_labels,
      // Array to store output active neurons in. This should be null if it is
      // not desired for the output values to be returned or if the output is
      // dense.
      uint32_t* output_active_neurons,
      // Array to store output activations in, will not return activations if
      // this is null
      float* output_activations,
      // Use sparse inference
      bool use_sparse_inference = false,
      // Metrics to compute
      const std::vector<std::string>& metric_names = {},
      // Restrict printouts
      bool verbose = true,
      // Limit the number of batches used in the dataset
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max());


inline void processTestBatch(
    const bolt::BoltBatch& batch_inputs, BoltBatch& outputs,
    const BoltBatch* batch_labels, uint32_t* output_active_neurons,
    float* output_activations, uint64_t inference_output_dim,
    MetricAggregator& metrics, bool compute_metrics);


  // Distributed Functions
  uint32_t initTrainDistributed(
      std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>>& train_data,
      const dataset::BoltDatasetPtr& train_labels,
      // Clang tidy is disabled for this line because it wants to pass by
      // reference, but shared_ptrs should not be passed by reference
      uint32_t rehash, uint32_t rebuild, bool verbose);

  void calculateGradientDistributed(uint32_t batch,
                                    const LossFunction& loss_fn);

  void updateParametersDistributed(float learning_rate);

  uint32_t getInferenceOutputDim(bool use_sparse_inference) const final;

    void forward(uint32_t batch_index, const bolt::BoltBatch& inputs,
      BoltVector& output, const BoltVector* labels) final {
        DistributedNetwork.forward(batch_index, inputs, output, labels);
      };

  void backpropagate(uint32_t batch_index, bolt::BoltBatch& inputs,
      BoltVector& output) final{
        DistributedNetwork.backpropagate(batch_index, inputs, output);
      };

  void updateParameters(float learning_rate, uint32_t iter) final {
    DistributedNetwork.updateParameters(learning_rate , iter);
  }

  void initializeNetworkState(uint32_t batch_size, bool use_sparsity) final{
    DistributedNetwork.initializeNetworkState(batch_size, use_sparsity);
  };


  BoltBatch getOutputs(uint32_t batch_size, bool use_sparsity) final {
    return DistributedNetwork.getOutputs(batch_size, use_sparsity);
  }

  uint32_t getOutputDim() const final;

  uint32_t numLayers() const;

  float* getLayerData(uint32_t layer_index, get_type type);

  void setLayerData(uint32_t layer_index, const float* data, set_type type);

  uint32_t getDim(uint32_t layer_index) const;

  uint32_t getInputDim() const;

  void reBuildHashFunctions() final {
    DistributedNetwork.reBuildHashFunctions();
  }
  

  void buildHashTables() final {
    DistributedNetwork.buildHashTables();
  }
  void setShallow(bool shallow) final {
    (void)shallow;
    throw thirdai::exceptions::NotImplemented(
        "Warning: setShallow not implemented for DLRM;");
  }

  void setShallowSave(bool shallow) final {
    (void)shallow;
    throw thirdai::exceptions::NotImplemented(
        "Warning: setShallowSave not implemented for DLRM;");
  }

  void setLayerSparsity(uint32_t layer_index, float sparsity, uint32_t hash_seed, uint32_t shuffle_seed) {
    DistributedNetwork.checkLayerIndex(layer_index);
    DistributedNetwork._layers.at(layer_index)->setSparsity(sparsity, hash_seed, shuffle_seed);
  }

  bool anyLayerShallow() final { return false; }
  
  BoltBatch _outputs;
  FullyConnectedNetwork DistributedNetwork;

 protected:
  static uint32_t getRehashBatchDistributed(uint32_t rehash, uint32_t batch_size,
                                     uint32_t data_len);

  static uint32_t getRebuildBatchDistributed(uint32_t rebuild, uint32_t batch_size,
                                      uint32_t data_len);

  uint32_t _batch_iter;

 private:
  
  uint32_t _epoch_count;
  uint32_t _rebuild_batch;
  uint32_t _rehash_batch;
  std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>> _train_data;
  dataset::BoltDatasetPtr _train_labels;



};

}  // namespace thirdai::bolt
