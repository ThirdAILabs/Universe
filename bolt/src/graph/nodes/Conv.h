#pragma once

#include <bolt/src/graph/Node.h>

namespace thirdai::bolt {

class ConvNode final : public Node,
                       public std::enable_shared_from_this<ConvNode> {
 private:
  ConvNode() {}

 public:
  static std::shared_ptr<ConvNode> make() {
    return std::make_shared<ConvNode>();
  }

  uint32_t outputDim() const final {
    NodeState node_state = getState();
    if (node_state == NodeState::Constructed ||
        node_state == NodeState::PredecessorsSet) {
      return _config->getDim();
    }
    return _layer->getDim();
  }

  bool isInputNode() const final { return false; }

  
};

}  // namespace thirdai::bolt