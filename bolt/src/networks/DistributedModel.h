#pragma once

#include <wrappers/src/LicenseWrapper.h>
#include <cereal/types/vector.hpp>
#include "Model.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
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

class DistributedModel : Model<bolt::BoltBatch> {
 public:
  enum GetType {
    GET_WEIGHTS,
    GET_BIASES,
    GET_WEIGHT_GRADIENTS,
    GET_BIASES_GRADIENTS
  };

  enum SetType {
    SET_WEIGHTS,
    SET_BIASES,
    SET_WEIGHTS_GRADIENTS,
    SET_BIASES_GRADIENTS
  };

  DistributedModel(SequentialConfigList configs, uint64_t input_dim)
      : DistributedNetwork(std::move(configs), input_dim, true),
        _batch_iter(0),
        _epoch_count(0),
        _rebuild_batch(0),
        _rehash_batch(0),
        _train_data(nullptr),
        _train_labels(nullptr) {
  }

  // This function is inherently calling predict in model.h
  // So, see model.h for more info
  InferenceMetricData predictSingleNode(
      const std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>>&test_data,
      const dataset::BoltDatasetPtr& test_labels,
      uint32_t* output_active_neurons,
      float* output_activations,
      bool use_sparse_inference = false,
      const std::vector<std::string>& metric_names = {},
      bool verbose = true,
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max());

  // Distributed Functions
  uint32_t initTrainSingleNode(
      std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>>& train_data,
      const dataset::BoltDatasetPtr& train_labels,
      uint32_t rehash, uint32_t rebuild, bool verbose);
  
/*
 * This function calculates the gradient using the train_data of
 * particular batch(provided using batch_no) and for a particular
 * loss_function.
 */
  void calculateGradientSingleNode(uint32_t batch, const LossFunction& loss_fn);

/*
 * This function updates the parameters for the neural network
 * using the updated gradients.
 * Right now, the updates are dense meaning that every parameter
 * is getting updated irrespective of type of training(dense or sparse)
 */
  void updateParametersSingleNode(float learning_rate);

  uint32_t getInferenceOutputDim(bool use_sparse_inference) const final;

  void forward(uint32_t batch_index, const bolt::BoltBatch& inputs,
               BoltVector& output, const BoltVector* labels) final {
    DistributedNetwork.forward(batch_index, inputs, output, labels);
  };

  void backpropagate(uint32_t batch_index, bolt::BoltBatch& inputs,
                     BoltVector& output) final {
    DistributedNetwork.backpropagate(batch_index, inputs, output);
  };

  void updateParameters(float learning_rate, uint32_t iter) final {
    DistributedNetwork.updateParameters(learning_rate, iter);
  }

  void initializeNetworkState(uint32_t batch_size, bool use_sparsity) final {
    DistributedNetwork.initializeNetworkState(batch_size, use_sparsity);
  };

  BoltBatch getOutputs(uint32_t batch_size, bool use_sparsity) final {
    return DistributedNetwork.getOutputs(batch_size, use_sparsity);
  }

  uint32_t getOutputDim() const final;

  uint32_t numLayers() const;

  float* getLayerData(uint32_t layer_index, GetType type);

  void setLayerData(uint32_t layer_index, const float* data, SetType type);

  uint32_t getDim(uint32_t layer_index) const;

  uint32_t getInputDim() const;

  void reBuildHashFunctions() final {
    DistributedNetwork.reBuildHashFunctions();
  }

  void buildHashTables() final { DistributedNetwork.buildHashTables(); }
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

  bool anyLayerShallow() final { return false; }

  void setRandomSeed(uint32_t random_seed) const;

  void freezeHashTables() { DistributedNetwork.freezeHashTables(); }
  // output  needed to be global variable because three
  // different function calls are using the same variable
  BoltBatch _outputs;

  FullyConnectedNetwork DistributedNetwork;

 protected:
  uint32_t _batch_iter;

 private:
  uint32_t _epoch_count;
  uint32_t _rebuild_batch;
  uint32_t _rehash_batch;
  std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>> _train_data;
  dataset::BoltDatasetPtr _train_labels;
};

}  // namespace thirdai::bolt
