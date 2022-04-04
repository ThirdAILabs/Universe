#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <algorithm>
#include <atomic>
#include <vector>
#include <cmath>

namespace thirdai::bolt {

/**
 * Helper function for incrementing an atomic float.
 * This is because there is no specialization for atomic floats prior to C++20 
 * and we were using C++17 at the time that this function was written.
 */
inline void incrementAtomicFloat(std::atomic<float>& atomic_float, float increment) {
  float current_value = atomic_float.load(std::memory_order_relaxed);
  while (!atomic_float.compare_exchange_weak(
    current_value, 
    current_value + increment, 
    std::memory_order_relaxed)) {};
}

/**
 * Implementation of the helper function below that will be compiled for any combination 
 * of sparse and dense vectors to minimize boolean checking.
 */
template<bool OUTPUT_DENSE, bool LABEL_DENSE>
inline void visitActiveNeuronsImpl(const BoltVector& output, const BoltVector& labels, 
                                   const std::function<void(float, float)>& process_elem_pair) {
  assert(!OUTPUT_DENSE || output.active_neurons == nullptr);
  assert(!LABEL_DENSE || labels.active_neurons == nullptr);
  if (OUTPUT_DENSE && LABEL_DENSE) {
    assert(output.len == labels.len);
  }

  std::vector<bool> labels_positions_touched(labels.len);
  for (uint32_t i = 0; i < output.len; i++) {
    uint32_t active_neuron = OUTPUT_DENSE ? i : output.active_neurons[i];
    float label_val;
    if (LABEL_DENSE) {
      label_val = labels.activations[active_neuron];
      labels_positions_touched[active_neuron] = true;
    } else {
      const uint32_t* label_start = labels.active_neurons;
      const uint32_t* label_end = labels.active_neurons + labels.len;
      const uint32_t* itr = std::find(label_start, label_end, active_neuron);
      if (itr == label_end) {
        label_val = 0.0;
      } else {
        size_t pos = std::distance(label_start, itr);
        label_val = labels.activations[pos];
        labels_positions_touched[pos] = true;
      }
    }
    process_elem_pair(label_val, output.activations[i]);
  }

  // Also consider neurons that are active in label but not in output.
  for (uint32_t i = 0; i < labels.len; i++) {
    if (!labels_positions_touched[i]) {
      process_elem_pair(labels.activations[i], /* output_val = */ 0.0);
    }
  }
}

/** 
 * Helper function that iterates through pairs of activations for the same neuron 
 * in output and labels for each neuron that is active in either the output, the label,
 * or both. This is particularly useful for regression metrics.
 *
 * Redirects function call to the appropriate templated implementation.
 */
inline void visitActiveNeurons(const BoltVector& output, const BoltVector& labels, 
                                const std::function<void(float, float)>& process_elem_pair) {
  if (output.isDense()) {
    if (labels.isDense()) {
      visitActiveNeuronsImpl<true, true>(output, labels, process_elem_pair);
    } else {
      visitActiveNeuronsImpl<true, false>(output, labels, process_elem_pair);
    }
  } else {
    if (labels.isDense()) {
      visitActiveNeuronsImpl<false, true>(output, labels, process_elem_pair);
    } else {
      visitActiveNeuronsImpl<false, false>(output, labels, process_elem_pair);
    }
  }
}

}  // namespace thirdai::bolt