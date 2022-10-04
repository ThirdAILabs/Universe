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
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

// Forward declare these clases so they can be a friend
// TODO(Josh): Clean this up when we refactor the dataset pipeline, there should
// eventually be just one wrapper here
class DistributedTabularTrainingWrapper;
class DistributedInMemoryTrainingWrapper;
class DistributedTrainingWrapper;

class BoltGraph {
  friend class DistributedTabularTrainingWrapper;
  friend class DistributedInMemoryTrainingWrapper;
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
        _updates(0) {
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

  MetricData train(const std::vector<dataset::BoltDatasetPtr>& train_data,
                   const dataset::BoltDatasetPtr& train_labels,
                   const TrainConfig& train_config);

  InferenceResult predict(const std::vector<dataset::BoltDatasetPtr>& test_data,
                          const dataset::BoltDatasetPtr& test_labels,
                          const PredictConfig& predict_config);

  std::pair<std::optional<std::vector<uint32_t>>, std::vector<float>>
  getInputGradientSingle(
      std::vector<BoltVector>&& input_data,
      bool explain_prediction_using_highest_activation = true,
      std::optional<uint32_t> neuron_to_explain = std::nullopt);

  BoltVector predictSingle(std::vector<BoltVector>&& test_data,
                           bool use_sparse_inference);

  BoltBatch predictSingleBatch(std::vector<BoltBatch>&& test_data,
                               bool use_sparse_inference);

  BoltVector getLabelVectorExplainPrediction(
      uint32_t vec_id, bool explain_prediction_using_highest_activation);

  BoltVector getLabelVectorNeuronsToExplain(uint32_t required_index,
                                            uint32_t vec_id);

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

 private:
  // Private constructor for cereal.
  BoltGraph() { thirdai::licensing::LicenseWrapper::checkLicense(); }

  void processTrainingBatch(const BoltBatch& batch_labels,
                            MetricAggregator& metrics);

  void processInferenceBatch(uint64_t batch_size, const BoltBatch* batch_labels,
                             MetricAggregator& metrics);

  void processOutputCallback(
      const std::optional<std::function<void(const BoltVector&)>>&
          output_callback,
      uint32_t batch_size);

  // Computes the forward pass through the graph.
  void forward(uint32_t vec_index, const BoltVector* labels);

  // Computes the backward pass through the graph.
  void backpropagate(uint32_t vec_index);

  void prepareToProcessBatches(uint32_t batch_size, bool use_sparsity);

  void cleanupAfterBatchProcessing();

  void updateParameters(float learning_rate, uint32_t batch_cnt);

  void resetOutputGradients(uint32_t vec_index);

  void updateParametersAndSampling(float learning_rate,
                                   uint32_t rebuild_hash_tables_batch,
                                   uint32_t reconstruct_hash_functions_batch);

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

  // This function makes sure all layers in the graph are prepard for
  // distributed training. This chiefly is relevant during paramemeter updates,
  // when a layer needs to know that it cannot rely on its own tracking of which
  // neurons were activated (since the gradient will also be aggregated from
  // other machines), and so it should do a dense parameter update no matter
  // what.
  void enableDistributedTraining();

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
};

using BoltGraphPtr = std::shared_ptr<BoltGraph>;

}  // namespace thirdai::bolt
