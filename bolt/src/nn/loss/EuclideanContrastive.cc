#include "EuclideanContrastive.h"
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/src/BoltVectorUtils.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace thirdai::bolt::nn::loss {

Contrastive::Contrastive(autograd::ComputationPtr output_1,
                         autograd::ComputationPtr output_2,
                         autograd::ComputationPtr labels,
                         float dissimilar_cutoff_margin)
    : _output_1(std::move(output_1)),
      _output_2(std::move(output_2)),
      _labels(std::move(labels)),
      _dissimilar_cutoff_margin(dissimilar_cutoff_margin) {
  if (dissimilar_cutoff_margin <= 0) {
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

std::shared_ptr<Contrastive> Contrastive::make(
    autograd::ComputationPtr output_1, autograd::ComputationPtr output_2,
    autograd::ComputationPtr labels, float dissimilar_cutoff_margin) {
  return std::make_shared<Contrastive>(output_1, output_2, labels,
                                       dissimilar_cutoff_margin);
}

void Contrastive::gradients(uint32_t index_in_batch,
                            uint32_t batch_size) const {
  (void)batch_size;
  float label = _labels->tensor()->getVector(index_in_batch).activations[0];
  auto& vec_1 = _output_1->tensor()->getVector(index_in_batch);
  auto& vec_2 = _output_2->tensor()->getVector(index_in_batch);

  float euclidean_distance =
      std::sqrt(euclideanDistanceSquared(index_in_batch));
  float dissimilar_multiplier =
      _dissimilar_cutoff_margin < euclidean_distance
          ? 0
          : (_dissimilar_cutoff_margin - euclidean_distance) /
                euclidean_distance;

  bolt_vector::visitPair(
      vec_1, vec_2,
      [&vec_1, &vec_2, label, dissimilar_multiplier](
          FoundActiveNeuron neuron_1, FoundActiveNeuron neuron_2) {
        float update = label * (neuron_1.activation - neuron_2.activation) +
                       (1 - label) * dissimilar_multiplier *
                           (neuron_2.activation - neuron_1.activation);
        if (neuron_1.pos) {
          vec_1.gradients[neuron_1.pos.value()] += update;
        }
        if (neuron_2.pos) {
          vec_2.gradients[neuron_2.pos.value()] -= update;
        }
      });
}

float Contrastive::loss(uint32_t index_in_batch) const {
  float label = _labels->tensor()->getVector(index_in_batch).activations[0];

  float euclidean_distance_squared = euclideanDistanceSquared(index_in_batch);
  float euclidean_distance = std::sqrt(euclidean_distance_squared);
  float cutoff_distance =
      std::max<float>(0, _dissimilar_cutoff_margin - euclidean_distance);
  float cutoff_distance_squared = cutoff_distance * cutoff_distance;

  return (1 - label) * 0.5 * euclidean_distance_squared +
         label * cutoff_distance_squared;
}

autograd::ComputationList Contrastive::outputsUsed() const {
  return {_output_1, _output_2};
}

autograd::ComputationList Contrastive::labels() const { return {_labels}; }

float Contrastive::euclideanDistanceSquared(uint32_t index_in_batch) const {
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

}  // namespace thirdai::bolt::nn::loss