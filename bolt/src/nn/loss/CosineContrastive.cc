#include "CosineContrastive.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/src/BoltVectorUtils.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace thirdai::bolt {

CosineContrastive::CosineContrastive(ComputationPtr output_1,
                                     ComputationPtr output_2,
                                     ComputationPtr labels,
                                     float dissimilar_cutoff_distance)
    : _output_1(std::move(output_1)),
      _output_2(std::move(output_2)),
      _labels(std::move(labels)),
      _dissimilar_cutoff_distance(dissimilar_cutoff_distance) {
  if (dissimilar_cutoff_distance <= 0) {
    throw std::invalid_argument(
        "The cutoff margin for the distance between dissimilar points must be "
        "greater than 0.");
  }
  if (_labels->dim() != 2) {
    throw std::invalid_argument(
        "The dimension of the labels for contrastive loss must equal 2, but "
        "found labels with dimension " +
        std::to_string(_labels->dim()));
  }
  if (_output_1->dim() != _output_2->dim()) {
    throw std::invalid_argument(
        "The dimension of the both outputs for contrastive loss must be the "
        "same, but found output 1 with dimension " +
        std::to_string(_output_1->dim()) + " and output 2 with dimension " +
        std::to_string(_output_2->dim()));
  }
}

std::shared_ptr<CosineContrastive> CosineContrastive::make(
    const ComputationPtr& output_1, const ComputationPtr& output_2,
    const ComputationPtr& labels, float dissimilar_cutoff_distance) {
  return std::make_shared<CosineContrastive>(output_1, output_2, labels,
                                             dissimilar_cutoff_distance);
}

void CosineContrastive::gradients(uint32_t index_in_batch,
                                  uint32_t batch_size) const {
  // See bolt/src/nn/derivations/CosineContrastive.md for this derivation.

  (void)batch_size;
  float label = _labels->tensor()->getVector(index_in_batch).active_neurons[0];
  auto& vec_1 = _output_1->tensor()->getVector(index_in_batch);
  auto& vec_2 = _output_2->tensor()->getVector(index_in_batch);

  float sim = cosineSim(index_in_batch);
  float cosine_distance = 1 - sim;
  // If the euclidean distance between points is 0, they were likely the same
  // input, or the network is in a degenerative state. Either way, the gradient
  // will be nan or inf, and we don't want this so we treat it as a NOOP.
  if (cosine_distance == 0) {
    return;
  }

  float umag = magnitude(vec_1);
  float vmag = magnitude(vec_2);
  float sim_over_umag_squared = sim / (umag * umag);
  float umag_vmag = umag * vmag;

  float multiplier =
      ((cosine_distance * label) -
       ((1 - label) *
        std::max<float>(_dissimilar_cutoff_distance - cosine_distance, 0))) /
      batch_size;

  bolt_vector::visitPair(
      vec_1, vec_2,
      [&vec_1, &vec_2, sim_over_umag_squared, umag_vmag, multiplier](
          FoundActiveNeuron neuron_1, FoundActiveNeuron neuron_2) {
        float derivative = (neuron_1.activation * sim_over_umag_squared) -
                           (neuron_2.activation / umag_vmag);
        float update = derivative * multiplier;

        if (neuron_1.pos) {
          vec_1.gradients[neuron_1.pos.value()] -= update;
        }
        if (neuron_2.pos) {
          vec_2.gradients[neuron_2.pos.value()] += update;
        }
      });
}

float CosineContrastive::loss(uint32_t index_in_batch) const {
  float label = _labels->tensor()->getVector(index_in_batch).active_neurons[0];

  float cosine_distance = 1 - cosineSim(index_in_batch);

  // If the cosine distance between two points is 0, they were likely the same
  // input, or the network is in a degenerative state. Either way, the gradient
  // will be nan or inf, and we don't want this so we treat it as not
  // contributing to the loss.
  if (cosine_distance == 0) {
    return 0;
  }
  float cosine_distance_squared = cosine_distance * cosine_distance;
  float cutoff_distance =
      std::max<float>(0, _dissimilar_cutoff_distance - cosine_distance);
  float cutoff_distance_squared = cutoff_distance * cutoff_distance;

  return label * 0.5 * cosine_distance_squared +
         (1 - label) * 0.5 * cutoff_distance_squared;
}

ComputationList CosineContrastive::outputsUsed() const {
  return {_output_1, _output_2};
}

ComputationList CosineContrastive::labels() const { return {_labels}; }

float CosineContrastive::cosineSim(uint32_t index_in_batch) const {
  auto& a = _output_1->tensor()->getVector(index_in_batch);
  auto& b = _output_2->tensor()->getVector(index_in_batch);

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

  return sim;
}

float CosineContrastive::magnitude(const BoltVector& a) {
  float squared = 0.0;
  for (size_t i = 0; i < a.len; i++) {
    squared += (a.activations[i] * a.activations[i]);
  }
  return std::sqrt(squared);
}

float CosineContrastive::denseDenseSim(const BoltVector& a,
                                       const BoltVector& b) {
  float a_dot_b = 0.0;
  for (size_t i = 0; i < a.len; i++) {
    a_dot_b += a.activations[i] * b.activations[i];
  }

  return a_dot_b / (magnitude(a) * magnitude(b));
}

float CosineContrastive::denseSparseSim(const BoltVector& a,
                                        const BoltVector& b) {
  float a_dot_b = 0.0;
  for (size_t i = 0; i < b.len; i++) {
    a_dot_b += a.activations[b.active_neurons[i]] * b.activations[i];
  }
  return a_dot_b / (magnitude(a) * magnitude(b));
}

float CosineContrastive::sparseSparseSim(const BoltVector& a,
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

template <class Archive>
void CosineContrastive::serialize(Archive& archive) {
  archive(cereal::base_class<Loss>(this), _output_1, _output_2, _labels,
          _dissimilar_cutoff_distance);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::CosineContrastive,
                               "thirdai::bolt::nn::loss::CosineContrastive")
