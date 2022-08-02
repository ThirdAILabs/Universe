#pragma once

#include <wrappers/src/LicenseWrapper.h>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include "DatasetContext.h"
#include "ExecutionConfig.h"
#include "InferenceOutputTracker.h"
#include "Node.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <dataset/src/batch_types/BoltTokenBatch.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

using GraphCallback = std::function<void()>;

class BoltGraph {
 public:
  /*
    The graph is constructed with a list of input layers, the order of these
    input layers is used to define how training/test inputs are mapped to the
    specific layers. Using the output node the graph can be traversed backwards
    to discover a reverse ordering in which to execute the layers.
   */
  BoltGraph(std::vector<InputPtr> inputs, NodePtr output)
      : BoltGraph(std::move(inputs), /* token-inputs= */ {},
                  std::move(output)) {
    thirdai::licensing::LicenseWrapper::checkLicense();
  }

  BoltGraph(std::vector<InputPtr> inputs,
            std::vector<TokenInputPtr> token_inputs, NodePtr output)
      : _output(std::move(output)),
        _inputs(std::move(inputs)),
        _token_inputs(std::move(token_inputs)),
        _epoch_count(0),
        _batch_cnt(0),
        _per_batch_callback(std::nullopt),
        _per_epoch_callback(std::nullopt) {
    thirdai::licensing::LicenseWrapper::checkLicense();
  }

  /*
    When the layers are initially defined the only have information about their
    own dimensions, parameters etc. During compile the layers can use the
    information from their predecessor(s) such as output dim to fully
    initialize their parameters. Additionally in this function checks are
    performed to ensure the graph is properly formatted. For instance if
    CategoricalCrossEntropy loss is used, then it can verify that the output
    layer has a softmax activation.
  */
  void compile(std::shared_ptr<LossFunction> loss, bool print_when_done = true);

  MetricData train(
      const std::vector<dataset::BoltDatasetPtr>& train_data,
      const std::vector<dataset::BoltTokenDatasetPtr>& train_tokens,
      const dataset::BoltDatasetPtr& train_labels,
      const TrainConfig& train_config);

  InferenceResult predict(
      const std::vector<dataset::BoltDatasetPtr>& test_data,
      const std::vector<dataset::BoltTokenDatasetPtr>& test_tokens,
      const dataset::BoltDatasetPtr& test_labels,
      const PredictConfig& predict_config,
      std::optional<std::function<void(const BoltVector&)>> output_callback =
          std::nullopt);

  BoltVector predictSingle(std::vector<BoltVector>&& test_data,
                           std::vector<std::vector<uint32_t>>&& test_tokens,
                           bool use_sparse_inference);

  std::vector<NodePtr> getNodeTraversalOrder() const {
    std::vector<NodePtr> nodes;
    nodes.insert(nodes.end(), _inputs.begin(), _inputs.end());
    nodes.insert(nodes.end(), _token_inputs.begin(), _token_inputs.end());
    nodes.insert(nodes.end(), _nodes.begin(), _nodes.end());

    return nodes;
  }

  void freezeHashTables(bool insert_labels_if_not_found);

  // This only saves the graph in the compiled state, that is any parameters and
  // graph structure are preserved, but any state related to train or predict is
  // discarded.
  void save(const std::string& filename);

  static std::unique_ptr<BoltGraph> load(const std::string& filename);

  std::string summarize(bool print, bool detailed) const;

  NodePtr getNodeByName(const std::string& node_name) const;

  void registerPerBatchCallback(GraphCallback callback) {
    _per_batch_callback = std::move(callback);
  }

  void registerPerEpochCallback(GraphCallback callback) {
    _per_epoch_callback = std::move(callback);
  }

 private:
  // Private constructor for cereal.
  BoltGraph() { thirdai::licensing::LicenseWrapper::checkLicense(); }

  void processTrainingBatch(const BoltBatch& batch_labels, float learning_rate,
                            MetricAggregator& metrics);

  void processInferenceBatch(uint64_t batch_size, const BoltBatch* batch_labels,
                             MetricAggregator& metrics);

  void processOutputCallback(
      std::optional<std::function<void(const BoltVector&)>> output_callback,
      uint32_t batch_size);

  // Computes the forward pass through the graph.
  void forward(uint32_t vec_index, const BoltVector* labels);

  // Computes the backward pass through the graph.
  void backpropagate(uint32_t vec_index);

  void prepareToProcessBatches(uint32_t batch_size, bool use_sparsity);

  void cleanupAfterBatchProcessing();

  void updateParameters(float learning_rate, uint32_t batch_cnt);

  void traverseGraph();

  std::unordered_map<NodePtr, int32_t> getSuccessorCounts() const;

  void verifyCanTrain(const DatasetContext& train_context);

  void verifyCanPredict(const DatasetContextBase& predict_context,
                        bool has_labels, bool returning_activations,
                        uint32_t num_metrics_tracked);

  void verifyInputForGraph(const DatasetContextBase& context);

  void verifyGraphProperties();

  void updateSampling(uint32_t rebuild_hash_tables_batch,
                      uint32_t reconstruct_hash_functions_batch);

  constexpr bool checkBatchInterval(uint32_t num_batches) const {
    return (_batch_cnt % num_batches) == (num_batches - 1);
  }

  void rebuildHashTables();

  void reconstructHashFunctions();

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  bool graphCompiled() const { return _loss != nullptr; }

  void perBatchCallback() {
    if (_per_batch_callback) {
      _per_batch_callback.value()();
    }
  }

  void perEpochCallback() {
    if (_per_epoch_callback) {
      _per_epoch_callback.value()();
    }
  }

  // List of nodes(layers) in the order in which they should be computed.
  std::vector<NodePtr> _nodes;

  // Output layer.
  NodePtr _output;

  // Input layers. When train is called, the ith input is fed into the ith input
  // layer.
  std::vector<InputPtr> _inputs;

  // Token input layers. Function similarly to the input layers but are specific
  // to nodes that require token inputs like the Embedding layer.
  std::vector<TokenInputPtr> _token_inputs;

  // List of the sparse layers in the graph. This is so that we can do
  // things like enable sparse inference, update hash tables, or update hash
  // functions.
  std::vector<std::shared_ptr<FullyConnectedLayer>>
      _internal_fully_connected_layers;

  std::shared_ptr<LossFunction> _loss;

  uint32_t _epoch_count;
  uint32_t _batch_cnt;

  std::optional<GraphCallback> _per_batch_callback;
  std::optional<GraphCallback> _per_epoch_callback;
};

using BoltGraphPtr = std::shared_ptr<BoltGraph>;

}  // namespace thirdai::bolt