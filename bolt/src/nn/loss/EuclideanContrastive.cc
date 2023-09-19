#include "EuclideanContrastive.h"
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

EuclideanContrastive::EuclideanContrastive(ComputationPtr output_1,
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
  if (_labels->dim() != 1) {
    throw std::invalid_argument(
        "The dimension of the labels for contrastive loss must equal 1, but "
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

std::shared_ptr<EuclideanContrastive> EuclideanContrastive::make(
    const ComputationPtr& output_1, const ComputationPtr& output_2,
    const ComputationPtr& labels, float dissimilar_cutoff_distance) {
  return std::make_shared<EuclideanContrastive>(output_1, output_2, labels,
                                                dissimilar_cutoff_distance);
}

void EuclideanContrastive::gradients(uint32_t index_in_batch,
                                     uint32_t batch_size) const {
  // See bolt/src/nn/derivations/EuclideanContrastive.md for this derivation.

  (void)batch_size;
  float label = _labels->tensor()->getVector(index_in_batch).activations[0];
  auto& vec_1 = _output_1->tensor()->getVector(index_in_batch);
  auto& vec_2 = _output_2->tensor()->getVector(index_in_batch);

  float euclidean_distance =
      std::sqrt(euclideanDistanceSquared(index_in_batch));
  // If the euclidean distance between points is 0, they were likely the same
  // input, or the network is in a degenerative state. Either way, the gradient
  // will be nan or inf, and we don't want this so we treat it as a NOOP.
  if (euclidean_distance == 0) {
    return;
  }
  float multiplier =
      (label -
       ((1 - label) *
        std::max<float>(_dissimilar_cutoff_distance - euclidean_distance, 0) /
        euclidean_distance)) /
      batch_size;

  bolt_vector::visitPair(
      vec_1, vec_2,
      [&vec_1, &vec_2, multiplier](FoundActiveNeuron neuron_1,
                                   FoundActiveNeuron neuron_2) {
        float update = multiplier * (neuron_1.activation - neuron_2.activation);
        if (neuron_1.pos) {
          vec_1.gradients[neuron_1.pos.value()] -= update;
        }
        if (neuron_2.pos) {
          vec_2.gradients[neuron_2.pos.value()] += update;
        }
      });
}

float EuclideanContrastive::loss(uint32_t index_in_batch) const {
  float label = _labels->tensor()->getVector(index_in_batch).activations[0];

  float euclidean_distance_squared = euclideanDistanceSquared(index_in_batch);
  // If the euclidean distance between points is 0, they were likely the same
  // input, or the network is in a degenerative state. Either way, the gradient
  // will be nan or inf, and we don't want this so we treat it as not
  // contributing to the loss.
  if (euclidean_distance_squared == 0) {
    return 0;
  }
  float euclidean_distance = std::sqrt(euclidean_distance_squared);
  float cutoff_distance =
      std::max<float>(0, _dissimilar_cutoff_distance - euclidean_distance);
  float cutoff_distance_squared = cutoff_distance * cutoff_distance;

  return label * 0.5 * euclidean_distance_squared +
         (1 - label) * 0.5 * cutoff_distance_squared;
}

ComputationList EuclideanContrastive::outputsUsed() const {
  return {_output_1, _output_2};
}

ComputationList EuclideanContrastive::labels() const { return {_labels}; }

float EuclideanContrastive::euclideanDistanceSquared(
    uint32_t index_in_batch) const {
  auto& vec_1 = _output_1->tensor()->getVector(index_in_batch);
  auto& vec_2 = _output_2->tensor()->getVector(index_in_batch);

  float euclidean_distance_squared = 0;
  bolt_vector::visitPair(
      vec_1, vec_2,
      [&euclidean_distance_squared](FoundActiveNeuron neuron_1,
                                    FoundActiveNeuron neuron_2) {
        float diff = (neuron_1.activation - neuron_2.activation);
        euclidean_distance_squared += diff * diff;
      });
  return euclidean_distance_squared;
}

template <class Archive>
void EuclideanContrastive::serialize(Archive& archive) {
  archive(cereal::base_class<Loss>(this), _output_1, _output_2, _labels,
          _dissimilar_cutoff_distance);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::EuclideanContrastive,
                               "thirdai::bolt::nn::loss::EuclideanContrastive")
