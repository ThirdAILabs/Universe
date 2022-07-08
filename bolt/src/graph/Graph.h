#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include "ExecutionConfig.h"
#include "Node.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

// Forward declerations
class InferenceOutputTracker;
class BoltGraph;

using InferenceResult = std::pair<InferenceMetricData, InferenceOutputTracker>;
using BoltGraphPtr = std::shared_ptr<BoltGraph>;

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
                  std::move(output)) {}

  BoltGraph(std::vector<InputPtr> inputs,
            std::vector<TokenInputPtr> token_inputs, NodePtr output)
      : _output(std::move(output)),
        _inputs(std::move(inputs)),
        _token_inputs(std::move(token_inputs)),
        _epoch_count(0),
        _batch_cnt(0) {}

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

  template <typename BATCH_T>
  MetricData train(
      // Train dataset
      std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& train_data,
      // Train labels
      const dataset::BoltDatasetPtr& train_labels,
      // Other train parameters
      const TrainConfig& train_config);

  template <typename BATCH_T>
  InferenceResult predict(
      // Test dataset
      const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& test_data,
      // Test labels
      const dataset::BoltDatasetPtr& test_labels,
      // Other prediction parameters
      const PredictConfig& predict_config);

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

 private:
  // Private constructor for cereal.
  BoltGraph() {}

  template <typename BATCH_T>
  void processTrainingBatch(BATCH_T& batch_inputs,
                            const BoltBatch& batch_labels, float learning_rate,
                            MetricAggregator& metrics);

  template <typename BATCH_T>
  void processInferenceBatch(BATCH_T& batch_inputs,
                             const BoltBatch* batch_labels,
                             MetricAggregator& metrics);

  template <typename BATCH_T>
  void setInputs(BATCH_T& batch_inputs);

  // Computes the forward pass through the graph.
  void forward(uint32_t vec_index, const BoltVector* labels);

  // Computes the backward pass through the graph.
  void backpropagate(uint32_t vec_index);

  void prepareToProcessBatches(uint32_t batch_size, bool use_sparsity);

  void cleanupAfterBatchProcessing();

  void updateParameters(float learning_rate, uint32_t batch_cnt);

  void traverseGraph();

  std::unordered_map<NodePtr, int32_t> getSuccessorCounts() const;

  template <typename BATCH_T>
  void verifyCanPredict(
      const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& test_data,
      bool has_labels, bool returning_activations,
      uint32_t num_metrics_tracked);

  template <typename BATCH_T>
  void verifyInputForGraph(
      const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& dataset);

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
};

// This class is NOT thread safe
class InferenceOutputTracker {
 public:
  // Should only be called after the output_node has been prepared for batch
  // processing
  InferenceOutputTracker(const NodePtr& output_node,
                         const PredictConfig& config,
                         uint32_t total_num_samples)
      : InferenceOutputTracker(
            /* num_nonzeros_per_sample = */ output_node->getOutputVector(0).len,
            /* num_samples = */ total_num_samples,
            /* save_output  = */ config.shouldReturnActivations(),
            /* output_sparse = */ !output_node->getOutputVector(0).isDense()) {}

  InferenceOutputTracker(uint32_t num_nonzeros_per_sample, uint32_t num_samples,
                         bool save_output, bool output_sparse)
      : _num_nonzeros_per_sample(num_nonzeros_per_sample),
        _num_samples(num_samples),
        _current_vec_index(0),
        _save_activations(save_output),
        _save_active_neurons(output_sparse && save_output) {
    // So the linteger won't complain in Release mode
    (void)_num_samples;
    uint64_t total_output_length = num_nonzeros_per_sample * num_samples;
    try {
      if (_save_activations) {
        _activations = std::vector<float>(total_output_length);
      } else {
        _activations = std::nullopt;
      }
      if (_save_active_neurons) {
        _active_neurons = std::vector<uint32_t>(total_output_length);
      } else {
        _active_neurons = std::nullopt;
      }
    } catch (std::bad_alloc& e) {
      throw std::invalid_argument(
          "Cannot allocate enough memory for inference output. Split the "
          "dataset into smaller batches and perform inference on each one, or "
          "change the PredictionConfig to do inference without getting the "
          "result.");
    }
  }

  void saveOutputBatch(const NodePtr& output_node, uint32_t batch_size) {
    for (uint32_t vec_id_in_batch = 0; vec_id_in_batch < batch_size;
         vec_id_in_batch++) {
      const auto& current_output_vec =
          output_node->getOutputVector(vec_id_in_batch);
      assert(current_output_vec.len == _num_nonzeros_per_sample);
      assert(_current_vec_index < _num_samples);

      if (_save_activations) {
        std::copy(
            current_output_vec.activations,
            current_output_vec.activations + _num_nonzeros_per_sample,
            &(_activations->at(_num_nonzeros_per_sample * _current_vec_index)));
      }

      if (_save_active_neurons) {
        assert(current_output_vec.active_neurons != nullptr);
        std::copy(current_output_vec.active_neurons,
                  current_output_vec.active_neurons + _num_nonzeros_per_sample,
                  &(_active_neurons->at(_num_nonzeros_per_sample *
                                        _current_vec_index)));
      }

      _current_vec_index++;
    }
  }

  // Returns a (possibly null) pointer to the saved activation data.
  // The pointer will be null if we did not save activations.
  const float* getNonowningActivationPointer() const {
    if (!_activations.has_value()) {
      return nullptr;
    }
    return _activations->data();
  }

  // Returns a (possibly null) pointer to the saved active neuron data.
  // The pointer will be null if we did not save activations or if the ouput
  // was dense.
  const uint32_t* getNonowningActiveNeuronPointer() const {
    if (!_active_neurons.has_value()) {
      return nullptr;
    }
    return _active_neurons->data();
  }

  uint32_t numNonzerosInOutput() const { return _num_nonzeros_per_sample; }

 private:
  uint64_t _num_nonzeros_per_sample;
  uint64_t _num_samples;
  uint64_t _current_vec_index;
  bool _save_activations;
  bool _save_active_neurons;
  std::optional<std::vector<float>> _activations;
  std::optional<std::vector<uint32_t>> _active_neurons;
};

}  // namespace thirdai::bolt