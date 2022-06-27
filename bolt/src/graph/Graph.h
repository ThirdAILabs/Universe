#pragma once

#include "Node.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

class Input;
using InputPtr = std::shared_ptr<Input>;

namespace tests {
class GraphTraversalTestFixture;
}  // namespace tests

class BoltGraph {
  friend class tests::GraphTraversalTestFixture;

 public:
  // The graph is constructed with a list of input layers, the order of these
  // input layers is used to define how training/test inputs are mapped to the
  // specific layers. Using the output node the graph can be traversed backwards
  // to discover a reverse ordering in which to execute the layers.
  BoltGraph(std::vector<InputPtr> inputs, NodePtr output)
      : _output(std::move(output)),
        _inputs(std::move(inputs)),
        _epoch_count(0),
        _batch_cnt(0) {}

  // When the layers are initially defined the only have information about their
  // own dimensions, parameters etc. During compile the layers can use the
  // information from their predecessor(s) such as output dim do fully
  // initialize their parameters. Note that successors could be added to nodes
  // as well if that is needed for certain layers to initialize. Additionally in
  // this function the different layers can preform different checks to make
  // sure that the network is properly formatted. For instance if
  // CategoricalCrossEntropy loss is used, then it can verify that the output
  // layer has a softmax activation.
  void compile(std::shared_ptr<LossFunction> loss);

  template <typename BATCH_T>
  MetricData train(
      // Train dataset
      std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& train_data,
      // Train labels
      const dataset::BoltDatasetPtr& train_labels,
      // Learning rate for training
      float learning_rate,
      // Number of training epochs
      uint32_t epochs,
      // Rehash, rebuild parameters for hash functions/tables
      std::optional<uint32_t> rebuild_hash_tables,
      std::optional<uint32_t> reconstruct_hash_functions,
      // Metrics to compute during training
      const std::vector<std::string>& metric_names = {},
      // Restrict printouts
      bool verbose = true);

  template <typename BATCH_T>
  InferenceMetricData predict(
      // Test dataset
      const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& test_data,
      // Test labels
      const dataset::BoltDatasetPtr& test_labels,
      // Use sparsity in inference
      bool use_sparsity,
      // Metrics to compute
      const std::vector<std::string>& metric_names = {},
      // Restrict printouts
      bool verbose = true,
      // Limit the number of batches used in the dataset
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max());

 private:
  template <typename BATCH_T>
  void processTrainingBatch(BATCH_T& batch_inputs,
                            const BoltBatch& batch_labels, float learning_rate,
                            MetricAggregator& metrics);

  template <typename BATCH_T>
  void processInferenceBatch(BATCH_T& batch_inputs,
                             const BoltBatch* batch_labels,
                             MetricAggregator& metrics, bool compute_metrics);

  // Computes the forward pass through the graph.
  void forward(uint32_t batch_index, const BoltVector* labels);

  // Computes the backward pass through the graph.
  void backpropagate(uint32_t batch_index);

  void prepareForBatchProcessing(uint32_t batch_size, bool use_sparsity);

  void updateParameters(float learning_rate, uint32_t batch_cnt);

  void traverseGraph();

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

  void rebuildHashFunctions();

  // List of nodes(layers) in the order in which they should be computed.
  std::vector<NodePtr> _nodes;

  // Output layer.
  NodePtr _output;

  // Input layers. When train is called, the ith input is fed into the ith input
  // layer.
  std::vector<InputPtr> _inputs;

  // List of the sparse layers in the graph. This is so that when we want to do
  // things like enable sparse inference, update hash tables, or update hash
  // functions.
  std::vector<std::shared_ptr<FullyConnectedLayer>> _sparse_layers;

  // The loss function the graph was compliled with.
  std::shared_ptr<LossFunction> _loss;

  uint32_t _epoch_count;
  uint32_t _batch_cnt;
};

using BoltGraphPtr = std::shared_ptr<BoltGraph>;

}  // namespace thirdai::bolt