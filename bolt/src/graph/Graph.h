#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

class Node;
using NodePtr = std::shared_ptr<Node>;

class Input;
using InputPtr = std::shared_ptr<Input>;

class GraphContext : public std::enable_shared_from_this<GraphContext> {
 public:
  GraphContext(std::vector<InputPtr> inputs, NodePtr output)
      : _output(std::move(output)), _inputs(std::move(inputs)) {}

  void compile(std::shared_ptr<LossFunction> loss);

  void forward(uint32_t batch_index, BoltBatch& inputs, BoltBatch& labels);

  void backward(uint32_t batch_index, BoltBatch& inputs);

  BoltVector& getLabels(uint32_t batch_index);

  std::shared_ptr<GraphContext> getPtr() { return shared_from_this(); }

 private:
  std::vector<NodePtr> _nodes;

  NodePtr _output;

  std::vector<InputPtr> _inputs;

  std::vector<std::shared_ptr<FullyConnectedLayer>> _sparse_layers;

  std::shared_ptr<LossFunction> _loss;
};

using GraphContextPtr = std::shared_ptr<GraphContext>;

}  // namespace thirdai::bolt