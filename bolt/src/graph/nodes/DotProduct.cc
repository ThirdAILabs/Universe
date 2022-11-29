#include "DotProduct.h"
#include <cereal/archives/binary.hpp>

namespace thirdai::bolt {

std::shared_ptr<DotProductNode> DotProductNode::setPredecessors(NodePtr lhs,
                                                                NodePtr rhs) {
  _lhs = std::move(lhs);
  _rhs = std::move(rhs);

  return shared_from_this();
}

void DotProductNode::prepareForBatchProcessingImpl(uint32_t batch_size,
                                                   bool use_sparsity) {
  (void)use_sparsity;
  _outputs = BoltBatch(/* dim= */ 1, /* batch_size=*/batch_size,
                       /* is_dense= */ true);
}

void DotProductNode::forwardImpl(uint32_t vec_index, const BoltVector* labels) {
  (void)labels;

  BoltVector& a = _lhs->getOutputVector(vec_index);
  BoltVector& b = _rhs->getOutputVector(vec_index);

  float output;
  if (a.isDense()) {
    if (b.isDense()) {
      output = denseDenseDotProduct(a, b);
    } else {
      output = denseSparseDotProduct(a, b);
    }
  } else {
    if (b.isDense()) {
      output = denseSparseDotProduct(b, a);
    } else {
      output = sparseSparseDotProduct(a, b);
    }
  }

  (*_outputs)[vec_index].activations[0] = 1 / (1 + std::exp(-output));
}

void DotProductNode::backpropagateImpl(uint32_t vec_index) {
  BoltVector& a = _lhs->getOutputVector(vec_index);
  BoltVector& b = _rhs->getOutputVector(vec_index);

  float grad = (*_outputs)[vec_index].gradients[0];

  if (a.isDense()) {
    if (b.isDense()) {
      denseDenseDotProductBackward(grad, a, b);
    } else {
      denseSparseDotProductBackward(grad, a, b);
    }
  } else {
    if (b.isDense()) {
      denseSparseDotProductBackward(grad, b, a);
    } else {
      sparseSparseDotProductBackward(grad, a, b);
    }
  }
}

void DotProductNode::summarizeImpl(std::stringstream& summary,
                                   bool detailed) const {
  (void)detailed;
  summary << "(" << _lhs->name() << ", " << _rhs->name() << ") -> " << name()
          << " (DotProduct)\n";
}

Node::NodeState DotProductNode::getState() const {
  if ((!_lhs && !_rhs) && !_compiled && !_outputs) {
    return NodeState::Constructed;
  }
  if ((_lhs && _rhs) && !_compiled && !_outputs) {
    return NodeState::PredecessorsSet;
  }
  if ((_lhs && _rhs) && _compiled && !_outputs) {
    return NodeState::Compiled;
  }
  if ((_lhs && _rhs) && _compiled && _outputs) {
    return NodeState::PreparedForBatchProcessing;
  }
  throw exceptions::NodeStateMachineError(
      "InputNode is in an invalid internal state");
}

float DotProductNode::denseDenseDotProduct(const BoltVector& a,
                                           const BoltVector& b) {
  assert(a.len == b.len);
  assert(a.isDense() && b.isDense());

  float total = 0;
  for (uint32_t i = 0; i < a.len; i++) {
    total += a.activations[i] * b.activations[i];
  }
  return total;
}

void DotProductNode::denseDenseDotProductBackward(float grad,
                                                  const BoltVector& a,
                                                  const BoltVector& b) {
  assert(a.len == b.len);
  assert(a.isDense() && b.isDense());

  for (uint32_t i = 0; i < a.len; i++) {
    a.gradients[i] += grad * b.activations[i];
    b.gradients[i] += grad * a.activations[i];
  }
}

float DotProductNode::denseSparseDotProduct(const BoltVector& dense_vec,
                                            const BoltVector& sparse_vec) {
  assert(dense_vec.isDense() && !sparse_vec.isDense());

  float total = 0;
  for (uint32_t i = 0; i < sparse_vec.len; i++) {
    assert(sparse_vec.active_neurons[i] < dense_vec.len);
    total += dense_vec.activations[sparse_vec.active_neurons[i]] *
             sparse_vec.activations[i];
  }
  return total;
}

void DotProductNode::denseSparseDotProductBackward(
    float grad, const BoltVector& dense_vec, const BoltVector& sparse_vec) {
  assert(dense_vec.isDense() && !sparse_vec.isDense());

  for (uint32_t i = 0; i < sparse_vec.len; i++) {
    assert(sparse_vec.active_neurons[i] < dense_vec.len);

    uint32_t active_neuron = sparse_vec.active_neurons[i];
    sparse_vec.gradients[i] += grad * dense_vec.activations[active_neuron];
    dense_vec.gradients[active_neuron] += grad * sparse_vec.activations[i];
  }
}

float DotProductNode::sparseSparseDotProduct(BoltVector& a, BoltVector& b) {
  assert(!a.isDense() && !b.isDense());

  a.sortActiveNeurons();
  b.sortActiveNeurons();

  float total = 0.0;

  applyFunctionToOverlappingNeurons(
      a, b, [&](uint32_t a_index, uint32_t b_index) {
        total += a.activations[a_index] * b.activations[b_index];
      });

  return total;
}

void DotProductNode::sparseSparseDotProductBackward(float grad, BoltVector& a,
                                                    BoltVector& b) {
  assert(!a.isDense() && !b.isDense());

  applyFunctionToOverlappingNeurons(
      a, b, [&](uint32_t a_index, uint32_t b_index) {
        a.gradients[a_index] += grad * b.activations[b_index];
        b.gradients[b_index] += grad * a.activations[a_index];
      });
}

void DotProductNode::applyFunctionToOverlappingNeurons(
    BoltVector& a, BoltVector& b,
    const std::function<void(uint32_t, uint32_t)>& func) {
  std::vector<uint32_t> overlapping_neurons;

  uint32_t a_index = 0;
  uint32_t b_index = 0;
  while (a_index < a.len && b_index < b.len) {
    if (a.active_neurons[a_index] == b.active_neurons[b_index]) {
      func(a_index, b_index);
      a_index++;
      b_index++;
    } else if (a.active_neurons[a_index] < b.active_neurons[b_index]) {
      a_index++;
    } else {
      b_index++;
    }
  }
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::DotProductNode)
