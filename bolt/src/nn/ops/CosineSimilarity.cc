#include "CosineSimilarity.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Map.h>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::bolt {

void CosineSimilarity::forward(const ComputationList& inputs, TensorPtr& output,
                               uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 2);

  const BoltVector& a = inputs.at(0)->tensor()->getVector(index_in_batch);
  const BoltVector& b = inputs.at(1)->tensor()->getVector(index_in_batch);
  BoltVector& out = output->getVector(index_in_batch);

  assert(out.len == 1);
  assert(out.isDense());

  float sim;
  if (a.isDense()) {
    if (b.isDense()) {
      sim = denseDenseSim(a, b);
    } else {
      sim = denseSparseSim(a, b);
    }
  } else {
    if (b.isDense()) {
      sim = denseSparseSim(b, a);
    } else {
      sim = sparseSparseSim(a, b);
    }
  }

  out.activations[0] = sim;
}

void CosineSimilarity::backpropagate(ComputationList& inputs, TensorPtr& output,
                                     uint32_t index_in_batch) {
  assert(inputs.size() == 2);

  BoltVector& a = inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& b = inputs.at(1)->tensor()->getVector(index_in_batch);
  const BoltVector& out = output->getVector(index_in_batch);

  assert(out.len == 1);
  assert(out.isDense());

  float cos_sim = out.activations[0];
  float grad = out.gradients[0];

  if (a.isDense()) {
    if (b.isDense()) {
      denseDenseBackprop(grad, cos_sim, a, b);
    } else {
      denseSparseBackprop(grad, cos_sim, a, b);
    }
  } else {
    if (b.isDense()) {
      denseSparseBackprop(grad, cos_sim, b, a);
    } else {
      sparseSparseBackprop(grad, cos_sim, a, b);
    }
  }
}

void CosineSimilarity::updateParameters(float learning_rate,
                                        uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

uint32_t CosineSimilarity::dim() const { return 1; }

std::optional<uint32_t> CosineSimilarity::nonzeros(
    const ComputationList& inputs, bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;

  return 1;
}

void CosineSimilarity::initOptimizer() {}

ComputationPtr CosineSimilarity::applyToInputs(const ComputationList& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "Expected CosineSimilarity op to have two inputs.");
  }
  return apply(inputs.at(0), inputs.at(1));
}

ar::ConstArchivePtr CosineSimilarity::toArchive(bool with_optimizer) const {
  (void)with_optimizer;

  auto map = baseArchive();
  map->set("type", ar::str(type()));
  return map;
}

std::shared_ptr<CosineSimilarity> CosineSimilarity::fromArchive(
    const ar::Archive& archive) {
  assertOpType(archive, type());

  auto op = CosineSimilarity::make();
  op->setName(archive.str("name"));
  return op;
}

void CosineSimilarity::summary(std::ostream& summary,
                               const ComputationList& inputs,
                               const Computation* output) const {
  summary << "CosineSimilarity(" << name() << "): " << inputs.at(0)->name()
          << ", " << inputs.at(1)->name() << " -> " << output->name();
}

ComputationPtr CosineSimilarity::apply(ComputationPtr lhs, ComputationPtr rhs) {
  if (lhs->dim() != rhs->dim()) {
    throw std::invalid_argument(
        "Cannot take cosine similarity between tensors with different "
        "dimensions.");
  }

  return Computation::make(shared_from_this(),
                           {std::move(lhs), std::move(rhs)});
}

float CosineSimilarity::magnitude(const BoltVector& a) {
  float squared = 0.0;
  for (size_t i = 0; i < a.len; i++) {
    squared += (a.activations[i] * a.activations[i]);
  }
  return std::sqrt(squared);
}

float CosineSimilarity::denseDenseSim(const BoltVector& a,
                                      const BoltVector& b) {
  float a_dot_b = 0.0;
  for (size_t i = 0; i < a.len; i++) {
    a_dot_b += a.activations[i] * b.activations[i];
  }

  return a_dot_b / (magnitude(a) * magnitude(b));
}

float CosineSimilarity::denseSparseSim(const BoltVector& a,
                                       const BoltVector& b) {
  float a_dot_b = 0.0;
  for (size_t i = 0; i < b.len; i++) {
    a_dot_b += a.activations[b.active_neurons[i]] * b.activations[i];
  }
  return a_dot_b / (magnitude(a) * magnitude(b));
}

float CosineSimilarity::sparseSparseSim(const BoltVector& a,
                                        const BoltVector& b) {
  std::unordered_map<uint32_t, float> b_map;
  for (size_t i = 0; i < b.len; i++) {
    b_map[b.active_neurons[i]] = b.activations[i];
  }

  float a_dot_b = 0.0;
  for (size_t i = 0; i < a.len; i++) {
    if (b_map.count(a.active_neurons[i])) {
      a_dot_b += a.activations[i] * b_map.at(a.active_neurons[i]);
    }
  }
  return a_dot_b / (magnitude(a) * magnitude(b));
}

// Computes the gradient for act given the matching activation from the other
// vector and other values.
float gradient(float grad, float act, float other_act, float mag,
               float other_mag, float cos_sim) {
  return grad * (other_act / (mag * other_mag) - act * cos_sim / (mag * mag));
}

void CosineSimilarity::denseDenseBackprop(float grad, float cos_sim,
                                          BoltVector& a, BoltVector& b) {
  float a_mag = magnitude(a);
  float b_mag = magnitude(b);

  for (size_t i = 0; i < a.len; i++) {
    float a_act = a.activations[i];
    float b_act = b.activations[i];
    a.gradients[i] += gradient(grad, a_act, b_act, a_mag, b_mag, cos_sim);
    b.gradients[i] += gradient(grad, b_act, a_act, b_mag, a_mag, cos_sim);
  }
}

void CosineSimilarity::denseSparseBackprop(float grad, float cos_sim,
                                           BoltVector& a, BoltVector& b) {
  float a_mag = magnitude(a);
  float b_mag = magnitude(b);

  for (size_t i = 0; i < b.len; i++) {
    float a_act = a.activations[b.active_neurons[i]];
    float b_act = b.activations[i];
    a.gradients[i] += gradient(grad, a_act, b_act, a_mag, b_mag, cos_sim);
    b.gradients[i] += gradient(grad, b_act, a_act, b_mag, a_mag, cos_sim);
  }
}

void CosineSimilarity::sparseSparseBackprop(float grad, float cos_sim,
                                            BoltVector& a, BoltVector& b) {
  std::unordered_map<uint32_t, size_t> b_map;
  for (size_t i = 0; i < b.len; i++) {
    b_map[b.active_neurons[i]] = i;
  }

  float a_mag = magnitude(a);
  float b_mag = magnitude(b);

  for (size_t i = 0; i < a.len; i++) {
    if (b_map.count(a.active_neurons[i])) {
      size_t j = b_map.at(a.active_neurons[i]);

      float a_act = a.activations[i];
      float b_act = b.activations[j];
      a.gradients[i] += gradient(grad, a_act, b_act, a_mag, b_mag, cos_sim);
      b.gradients[i] += gradient(grad, b_act, a_act, b_mag, a_mag, cos_sim);
    }
  }
}

template <class Archive>
void CosineSimilarity::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this));
}
}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::CosineSimilarity)