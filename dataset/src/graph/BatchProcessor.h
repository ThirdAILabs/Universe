#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/graph/Node.h>
#include <sys/types.h>
#include <algorithm>
#include <functional>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace thirdai::dataset::graph {

struct WorkflowOutput {
  WorkflowOutput(ProducerNodePtr<BoltVector> input_vector,
                 ProducerNodePtr<BoltVector> label_vector,
                 std::vector<SideEffectNodePtr> side_effects)
      : input_vector_producer(std::move(input_vector)),
        label_vector_producer(std::move(label_vector)),
        side_effect_nodes(std::move(side_effects)) {}

  ProducerNodePtr<BoltVector> input_vector_producer;
  ProducerNodePtr<BoltVector> label_vector_producer;
  std::vector<SideEffectNodePtr> side_effect_nodes;
};

class GraphBatchProcessor final : public BatchProcessor<BoltBatch, BoltBatch> {
 public:
  explicit GraphBatchProcessor(
      std::function<WorkflowOutput(const StringInputPtr&)> workflow)
      : _input_node(StringInput::make()) {
    auto output = workflow(_input_node);
    _input_vector_producer = output.input_vector_producer;
    _label_vector_producer = output.label_vector_producer;

    orderNodesByDependency(output);
  }

  std::tuple<BoltBatch, BoltBatch> createBatch(
      const std::vector<std::string>& rows) final {
    std::vector<BoltVector> input_vectors(rows.size());
    std::vector<BoltVector> label_vectors(rows.size());

    _input_node->allocateMemoryForBatchProduct(rows.size());
    for (auto& node : _nodes) {
      node->allocateMemoryForBatchProduct(rows.size());
    }

    _input_node->feed(rows);
    
#pragma omp parallel for default(none) \
    shared(rows, input_vectors, label_vectors)
    for (uint32_t row_idx = 0; row_idx < rows.size(); row_idx++) {
      for (auto& node : _nodes) {
        node->process(row_idx);
      }
      input_vectors[row_idx] = _input_vector_producer->getProduct(row_idx);
      label_vectors[row_idx] = _label_vector_producer->getProduct(row_idx);
    }

    return {
        BoltBatch(std::move(input_vectors)),
        BoltBatch(std::move(label_vectors)),
    };
  }

  void processHeader(const std::string& header) final { (void)header; }

  bool expectsHeader() const final { return false; }

  static std::vector<NodePtr> topLevelNodes(const WorkflowOutput& output) {
    std::vector<NodePtr> top_nodes;
    top_nodes.insert(top_nodes.begin(), output.side_effect_nodes.begin(),
                     output.side_effect_nodes.end());
    top_nodes.push_back(output.input_vector_producer);
    top_nodes.push_back(output.label_vector_producer);
    return top_nodes;
  }

  void orderNodesByDependency(const WorkflowOutput& workflow_output) {
    /*
      We want to build an ordering of nodes such that nodes always come after
      their dependencies; nodes must not depend on subsequent nodes.

      Visit a queued node only if all of its successors have been visited.
      If a dependency had been added, there must be a cycle.
    */

    std::queue<NodePtr> queue;
    std::unordered_map<NodePtr, uint32_t> times_visited;
    std::unordered_set<NodePtr> added;

    for (auto node : topLevelNodes(workflow_output)) {
      queue.push(std::move(node));
    }

    while (!queue.empty()) {
      auto next = queue.front();
      times_visited[next]++;
      queue.pop();

      // # times node appears at the front of the queue
      // equals # of successors.
      if (times_visited[next] < next->numSuccessors()) {
        continue;
      }

      for (auto pred : next->predecessors()) {
        if (added.count(pred)) {
          throw std::runtime_error(
              "Found a cycle in the data pipeline graph. Node: " +
              next->describe() + " Predecessor: " + pred->describe());
        }
        queue.push(std::move(pred));
      }
      _nodes.insert(_nodes.begin(), next);
      added.insert(next);
    }

    if (_nodes.front() != _input_node) {
      throw std::runtime_error(
          "Invalid data pipeline graph; root of graph is not input node.");
    }
  }

 private:
  std::vector<NodePtr> _nodes;
  StringInputPtr _input_node;
  ProducerNodePtr<BoltVector> _input_vector_producer;
  ProducerNodePtr<BoltVector> _label_vector_producer;
};

class ConverterNodes : public Node {

};

}  // namespace thirdai::dataset::graph