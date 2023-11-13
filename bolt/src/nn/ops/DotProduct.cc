#include "DotProduct.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/ArchiveMap.h>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::bolt {

void DotProduct::forward(const ComputationList& inputs, TensorPtr& output,
                         uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 2);

  const BoltVector& a = inputs.at(0)->tensor()->getVector(index_in_batch);
  const BoltVector& b = inputs.at(1)->tensor()->getVector(index_in_batch);
  BoltVector& out = output->getVector(index_in_batch);

  assert(out.len == 1);
  assert(out.isDense());

  float dot_product;
  if (a.isDense()) {
    if (b.isDense()) {
      dot_product = denseDenseDot(a, b);
    } else {
      dot_product = denseSparseDot(a, b);
    }
  } else {
    if (b.isDense()) {
      dot_product = denseSparseDot(b, a);
    } else {
      dot_product = sparseSparseDot(a, b);
    }
  }

  // Sigmoid to map to (0, 1)
  out.activations[0] = 1 / (1 + std::exp(-dot_product));
}

void DotProduct::backpropagate(ComputationList& inputs, TensorPtr& output,
                               uint32_t index_in_batch) {
  assert(inputs.size() == 2);

  BoltVector& a = inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& b = inputs.at(1)->tensor()->getVector(index_in_batch);
  const BoltVector& out = output->getVector(index_in_batch);

  assert(out.len == 1);
  assert(out.isDense());

  // In our code we combine the gradient of sigmoid with the gradient in binary
  // cross entropy loss. This simplifies the computations and makes it faster to
  // compute. This grad method assumes that binary cross entropy is being
  // applied to the output of the dot product op, thus the gradient for sigmoid
  // has already be taken care of here.
  float grad = out.gradients[0];

  if (a.isDense()) {
    if (b.isDense()) {
      denseDenseBackprop(grad, a, b);
    } else {
      denseSparseBackprop(grad, a, b);
    }
  } else {
    if (b.isDense()) {
      denseSparseBackprop(grad, b, a);
    } else {
      sparseSparseBackprop(grad, a, b);
    }
  }
}

void DotProduct::updateParameters(float learning_rate, uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

uint32_t DotProduct::dim() const { return 1; }

std::optional<uint32_t> DotProduct::nonzeros(const ComputationList& inputs,
                                             bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;

  return 1;
}

void DotProduct::initOptimizer() {}

ComputationPtr DotProduct::applyToInputs(const ComputationList& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument("Expected DotProduct op to have two inputs.");
  }
  return apply(inputs.at(0), inputs.at(1));
}

ar::ConstArchivePtr DotProduct::toArchive(bool with_optimizer) const {
  (void)with_optimizer;

  auto map = ar::ArchiveMap::make();
  map->set("name", ar::str(name()));
  map->set("type", ar::str("dot_product"));
  return map;
}

void DotProduct::summary(std::ostream& summary, const ComputationList& inputs,
                         const Computation* output) const {
  summary << "DotProduct(" << name() << "): " << inputs.at(0)->name() << ", "
          << inputs.at(1)->name() << " -> " << output->name();
}

ComputationPtr DotProduct::apply(ComputationPtr lhs, ComputationPtr rhs) {
  if (lhs->dim() != rhs->dim()) {
    throw std::invalid_argument(
        "Cannot take dot product between tensors with different dimensions.");
  }

  return Computation::make(shared_from_this(),
                           {std::move(lhs), std::move(rhs)});
}

float DotProduct::denseDenseDot(const BoltVector& a, const BoltVector& b) {
  float total = 0.0;
  for (size_t i = 0; i < a.len; i++) {
    total += a.activations[i] * b.activations[i];
  }

  return total;
}

float DotProduct::denseSparseDot(const BoltVector& a, const BoltVector& b) {
  float total = 0.0;
  for (size_t i = 0; i < b.len; i++) {
    total += a.activations[b.active_neurons[i]] * b.activations[i];
  }
  return total;
}

float DotProduct::sparseSparseDot(const BoltVector& a, const BoltVector& b) {
  std::unordered_map<uint32_t, float> b_map;
  for (size_t i = 0; i < b.len; i++) {
    b_map[b.active_neurons[i]] = b.activations[i];
  }

  float total = 0.0;
  for (size_t i = 0; i < a.len; i++) {
    if (b_map.count(a.active_neurons[i])) {
      total += a.activations[i] * b_map.at(a.active_neurons[i]);
    }
  }
  return total;
}

void DotProduct::denseDenseBackprop(float grad, BoltVector& a, BoltVector& b) {
  for (size_t i = 0; i < a.len; i++) {
    a.gradients[i] += grad * b.activations[i];
    b.gradients[i] += grad * a.activations[i];
  }
}

void DotProduct::denseSparseBackprop(float grad, BoltVector& a, BoltVector& b) {
  for (size_t i = 0; i < b.len; i++) {
    a.gradients[b.active_neurons[i]] += grad * b.activations[i];
    b.gradients[i] += grad * a.activations[b.active_neurons[i]];
  }
}

void DotProduct::sparseSparseBackprop(float grad, BoltVector& a,
                                      BoltVector& b) {
  std::unordered_map<uint32_t, size_t> b_map;
  for (size_t i = 0; i < b.len; i++) {
    b_map[b.active_neurons[i]] = i;
  }

  for (size_t i = 0; i < a.len; i++) {
    if (b_map.count(a.active_neurons[i])) {
      size_t j = b_map.at(a.active_neurons[i]);
      a.gradients[i] += grad * b.activations[j];
      b.gradients[j] += grad * a.activations[i];
    }
  }
}

template <class Archive>
void DotProduct::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this));
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::DotProduct)