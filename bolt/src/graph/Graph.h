#pragma once

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
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <bolt_vector/src/BoltVector.h>
#include <licensing/src/CheckLicense.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

class DistributedTrainingWrapper;

class BoltGraph {
  friend class DistributedTrainingWrapper;

 public:
  /*
    The graph is constructed with a list of input layers, the order of these
    input layers is used to define how training/test inputs are mapped to the
    specific layers. Using the output node the graph can be traversed backwards
    to discover a reverse ordering in which to execute the layers.
   */
  BoltGraph(std::vector<InputPtr> inputs, NodePtr output)
      : _output(std::move(output)),
        _inputs(std::move(inputs)),
        _epoch(0),
        _updates(0),
        _tracked_metric(nullptr) {
    thirdai::licensing::checkLicense();
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

  MetricData train(const std::vector<dataset::BoltDatasetPtr>& train_data,
                   const dataset::BoltDatasetPtr& train_labels,
                   const TrainConfig& train_config,
                   licensing::FinegrainedAccessToken token =
                       licensing::FinegrainedAccessToken());

  void trainOnBatch(std::vector<BoltBatch>&& inputs, const BoltBatch& labels,
                    float learning_rate, MetricAggregator& metrics,
                    uint32_t rebuild_hash_tables_interval,
                    uint32_t reconstruct_hash_functions_interval,
                    licensing::FinegrainedAccessToken token =
                        licensing::FinegrainedAccessToken());

  InferenceResult evaluate(
      const std::vector<dataset::BoltDatasetPtr>& test_data,
      const dataset::BoltDatasetPtr& test_labels,
      const EvalConfig& eval_config);

  std::pair<std::optional<std::vector<uint32_t>>, std::vector<float>>
  getInputGradientSingle(
      std::vector<BoltVector>&& input_data,
      bool explain_prediction_using_highest_activation = true,
      std::optional<uint32_t> neuron_to_explain = std::nullopt);

  BoltVector predictSingle(
      std::vector<BoltVector>&& test_data, bool use_sparse_inference,
      std::optional<std::string> output_node_name = std::nullopt);

  BoltBatch predictSingleBatch(std::vector<BoltBatch>&& test_data,
                               bool use_sparse_inference);

  std::vector<NodePtr> getNodeTraversalOrder() const {
    std::vector<NodePtr> nodes;
    nodes.insert(nodes.end(), _inputs.begin(), _inputs.end());
    nodes.insert(nodes.end(), _nodes.begin(), _nodes.end());

    return nodes;
  }

  void freezeHashTables(bool insert_labels_if_not_found);

  // This only saves the graph in the compiled state, that is any parameters and
  // graph structure are preserved, but any state related to train or predict is
  // discarded.
  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static BoltGraphPtr load(const std::string& filename);

  static BoltGraphPtr load_stream(std::istream& input_stream);

  std::string summarize(bool print, bool detailed) const;

  NodePtr getNodeByName(const std::string& node_name) const;

  uint32_t outputDim() const { return _output->outputDim(); }

  std::vector<NodePtr> getNodes() { return _nodes; }

  NodePtr output() { return _output; }

 private:
  // Private constructor for cereal.
  BoltGraph() { thirdai::licensing::checkLicense(); }

  void processTrainingBatch(const BoltBatch& batch_labels,
                            MetricAggregator& metrics);

  void logValidateAndSave(const TrainConfig& train_config,
                          MetricAggregator& train_metrics);

  void processEvaluationBatch(uint64_t batch_size,
                              const BoltBatch* batch_labels,
                              MetricAggregator& metrics, bool use_sparsity);

  void processOutputCallback(
      const std::optional<std::function<void(const BoltVector&)>>&
          output_callback,
      uint32_t batch_size);

  // Computes the forward pass through the graph.
  void forward(uint32_t vec_index, const BoltVector* labels);

  // Computes the backward pass through the graph.
  void backpropagate(uint32_t vec_index);

  void prepareToProcessBatch(uint32_t batch_size, bool use_sparsity);

  void updateParameters(float learning_rate, uint32_t batch_cnt);

  void resetOutputGradients(uint32_t vec_index);

  void updateParametersAndSampling(float learning_rate,
                                   uint32_t rebuild_hash_tables_batch,
                                   uint32_t reconstruct_hash_functions_batch);

  BoltVector getLabelVectorExplainPrediction(
      uint32_t vec_id, bool explain_prediction_using_highest_activation);

  BoltVector getLabelVectorNeuronsToExplain(uint32_t required_index,
                                            uint32_t vec_id);

  void traverseGraph();

  std::unordered_map<NodePtr, int32_t> getSuccessorCounts() const;

  void verifyCanTrain(const DatasetContext& train_context);

  void verifyCanGetInputGradientSingle(
      const DatasetContextBase& single_input_gradients_context,
      bool explain_prediction_using_highest_activation,
      uint32_t num_output_nonzeros);

  void verifyCanPredict(const DatasetContextBase& predict_context,
                        bool has_labels, bool returning_activations,
                        uint32_t num_metrics_tracked);

  void verifyInputForGraph(const DatasetContextBase& context);

  void verifyGraphProperties();

  void updateSampling(uint32_t rebuild_hash_tables_batch,
                      uint32_t reconstruct_hash_functions_batch);

  // This function prevents nodes from using sparse optimizations during
  // parameter updates. This is to make updateParameters work during distributed
  // training or disable the optimization in the few cases where they are not
  // beneficial.
  void disableSparseParameterUpdates();

  constexpr bool checkBatchInterval(uint32_t num_batches) const {
    return (_updates % num_batches) == (num_batches - 1);
  }

  void rebuildHashTables();

  void reconstructHashFunctions();

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  bool graphCompiled() const { return _loss != nullptr; }

  // List of nodes(layers) in the order in which they should be computed.
  std::vector<NodePtr> _nodes;

  // Output layer.
  NodePtr _output;

  // Input layers. When train is called, the ith input is fed into the ith input
  // layer.
  std::vector<InputPtr> _inputs;

  // List of the sparse layers in the graph. This is so that we can do
  // things like enable sparse inference, update hash tables, or update hash
  // functions.
  std::vector<std::shared_ptr<FullyConnectedLayer>>
      _internal_fully_connected_layers;

  std::shared_ptr<LossFunction> _loss;

  // TODO(blaise/david): Factor out _epoch and _updates and put
  // them in TrainState
  uint32_t _epoch;
  uint32_t _updates;

  class BatchProcessingState {
   public:
    BatchProcessingState() : _allocated_batch_size(0), _using_sparsity(false) {}

    BatchProcessingState(uint32_t batch_size, bool using_sparsity)
        : _allocated_batch_size(batch_size),
          _using_sparsity(using_sparsity),
          _optimizer_initialized(false) {}

    bool compatableWith(uint32_t batch_size, bool using_sparsity) const {
      return batch_size <= _allocated_batch_size &&
             using_sparsity == _using_sparsity;
    }

    bool isOptimizerInitialized() const { return _optimizer_initialized; }

    void markOptimizerInitialized() { _optimizer_initialized = true; }

   private:
    uint32_t _allocated_batch_size;
    bool _using_sparsity;
    bool _optimizer_initialized;
  };
  BatchProcessingState _batch_processing_state;

  // We need this value saved across training. Reset should be done by force.
  double _best_validation_metric;
  std::shared_ptr<Metric> _tracked_metric;
};

using BoltGraphPtr = std::shared_ptr<BoltGraph>;

}  // namespace thirdai::bolt
