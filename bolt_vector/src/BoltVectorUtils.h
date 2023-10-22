#pragma once

#include "BoltVector.h"
#include <fftw3.h>

namespace thirdai::bolt_vector {

// This function accepts two vectors of the same dimension and calls the passed
// in Visitor with one pair of FoundActiveNeurons for every neuron that
// is active in at least one of the vectors.
template <bool VEC_1_DENSE, bool VEC_2_DENSE, typename Visitor>
void templatedVisitPair(const BoltVector& vec_1, const BoltVector& vec_2,
                        Visitor visitor) {
  assert(VEC_1_DENSE == vec_1.isDense());
  assert(VEC_2_DENSE == vec_2.isDense());
  if constexpr (VEC_1_DENSE && VEC_2_DENSE) {
    assert(vec_1.len == vec_2.len);
  }
  static_assert(
      std::is_invocable<Visitor, FoundActiveNeuron, FoundActiveNeuron>::value,
      "Visitor must be a function that accepts two FoundActiveNeurons");

  if constexpr (VEC_1_DENSE || VEC_2_DENSE) {
    // We know that one of the vectors is dense, and we have as an assumption
    // that the vectors are the same dimension, so this is safe.
    uint32_t dim = std::max(vec_1.len, vec_2.len);
    for (uint32_t active_neuron = 0; active_neuron < dim; active_neuron++) {
      FoundActiveNeuron vec_1_found =
          vec_1.findActiveNeuron<VEC_1_DENSE>(active_neuron);
      FoundActiveNeuron vec_2_found =
          vec_2.findActiveNeuron<VEC_2_DENSE>(active_neuron);
      visitor(vec_1_found, vec_2_found);
    }
    return;
  }

  /**
   * If both are sparse then we need to iterate over the nonzeros from both
   * vectors. To avoid double visiting the overlapping neurons we only look at
   * overlapping neurons when iterating over the vec_1.
   */
  for (uint32_t vec_1_index = 0; vec_1_index < vec_1.len; vec_1_index++) {
    uint32_t active_neuron = vec_1.active_neurons[vec_1_index];
    float vec_1_activation = vec_1.activations[vec_1_index];
    FoundActiveNeuron vec_2_found =
        vec_2.findActiveNeuron<VEC_2_DENSE>(active_neuron);
    FoundActiveNeuron vec_1_found = {vec_1_index, vec_1_activation};
    visitor(vec_1_found, vec_2_found);
  }

  for (uint32_t vec_2_index = 0; vec_2_index < vec_2.len; vec_2_index++) {
    uint32_t active_neuron = vec_2.active_neurons[vec_2_index];
    FoundActiveNeuron vec_1_found =
        vec_1.findActiveNeuron<VEC_1_DENSE>(active_neuron);

    // Skip any neurons that were in vec_1 to avoid double visiting.
    if (!vec_1_found.pos.has_value()) {
      float vec_2_activation = vec_2.activations[vec_2_index];
      FoundActiveNeuron vec_2_found = {vec_2_index, vec_2_activation};
      // The activation is 0 since this isn't in the output active neurons.
      visitor(vec_1_found, vec_2_found);
    }
  }
}

static float* fftwSegmentRowMajorActivations(const BoltVector& base_vector, uint32_t rows, uint32_t columns){
  assert(rows * columns == base_vector.len);
  std::vector<BoltVector> segmented_vectors;
  float* input_data = static_cast<float*>(fftwf_malloc(columns * rows * sizeof(float)));
  for (uint32_t i = 0; i < rows; i++) {
    std::memcpy(&input_data[i * columns], base_vector.activations + i * columns, columns * sizeof(float));
  }
  return input_data;
}

static float* fftwSegmentRowMajorGradients(const BoltVector& base_vector, uint32_t rows, uint32_t columns){
  if(!base_vector.hasGradients()){
    throw std::runtime_error("This operation is not valid. Since, base_vector doesn't have gradients.");
  }
  assert(rows * columns == base_vector.len);
  std::vector<BoltVector> segmented_vectors;
  float* input_data = static_cast<float*>(fftwf_malloc(columns * rows * sizeof(float)));
  for (uint32_t i = 0; i < rows; i++) {
    std::memcpy(&input_data[i * columns], base_vector.gradients + i * columns, columns * sizeof(float));
  }
  return input_data;
}

template <typename Visitor>
void visitPair(const BoltVector& vec_1, const BoltVector& vec_2,
               Visitor visitor) {
  if (vec_1.isDense() && vec_2.isDense()) {
    return templatedVisitPair<true, true, Visitor>(vec_1, vec_2, visitor);
  }
  if (vec_1.isDense() && !vec_2.isDense()) {
    return templatedVisitPair<true, false, Visitor>(vec_1, vec_2, visitor);
  }
  if (!vec_1.isDense() && vec_2.isDense()) {
    return templatedVisitPair<false, true, Visitor>(vec_1, vec_2, visitor);
  }
  if (!vec_1.isDense() && !vec_2.isDense()) {
    return templatedVisitPair<false, false, Visitor>(vec_1, vec_2, visitor);
  }
}

}  // namespace thirdai::bolt_vector