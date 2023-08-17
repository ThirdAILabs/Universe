#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <type_traits>
#include <vector>

namespace thirdai::bolt_v1 {

class MetricUtilities {
 public:
  /**
   * Helper function for incrementing an atomic float.
   * This is because there is no specialization for atomic floats prior to C++20
   * and we were using C++17 at the time that this function was written.
   */
  static void incrementAtomicFloat(std::atomic<float>& atomic_float,
                                   float increment) {
    float current_value = atomic_float.load(std::memory_order_relaxed);
    while (!atomic_float.compare_exchange_weak(
        current_value, current_value + increment, std::memory_order_relaxed)) {
    };
  }

  /**
   * Helper function that iterates through all nonzero values in the output or
   * label vectors and applies a lambda to the corresponding pair of
   * (value_in_label_vector, value_in_output_vector).
   * Since this lambda takes in the values of vector entries, it accepts two
   * floats and does not return anything.
   *
   * This is particularly useful for calculating regression metrics.
   *
   * Implementation details:
   * - It redirects to the appropriate templated implementation.
   * - Lambda type is templated because this helps the compiler inline
   *   the lambda call.
   *   https://stackoverflow.com/questions/13722426/why-can-lambdas-be-better-optimized-by-the-compiler-than-plain-functions
   */
  template <typename PROCESS_ELEM_PAIR_LAMBDA_T>
  static void visitActiveNeurons(
      const BoltVector& output, const BoltVector& labels,
      PROCESS_ELEM_PAIR_LAMBDA_T process_elem_pair_lambda) {
    if (output.isDense()) {
      if (labels.isDense()) {
        visitActiveNeuronsImpl<true, true>(output, labels,
                                           process_elem_pair_lambda);
      } else {
        visitActiveNeuronsImpl<true, false>(output, labels,
                                            process_elem_pair_lambda);
      }
    } else {
      if (labels.isDense()) {
        visitActiveNeuronsImpl<false, true>(output, labels,
                                            process_elem_pair_lambda);
      } else {
        visitActiveNeuronsImpl<false, false>(output, labels,
                                             process_elem_pair_lambda);
      }
    }
  }

 private:
  /**
   * Implementation of the helper function above that will be compiled for any
   * combination of sparse and dense vectors to minimize boolean checking.
   *
   * The lambda takes in two floats and does not return anything.
   * Lambda type is templated because this helps the compiler inline
   * the lambda call.
   * https://stackoverflow.com/questions/13722426/why-can-lambdas-be-better-optimized-by-the-compiler-than-plain-functions
   */
  template <bool OUTPUT_DENSE, bool LABEL_DENSE,
            typename PROCESS_ELEM_PAIR_LAMBDA_T>
  static void visitActiveNeuronsImpl(
      const BoltVector& output, const BoltVector& labels,
      PROCESS_ELEM_PAIR_LAMBDA_T process_elem_pair) {
    // Asserts that the lambda takes in 2 floats and does not return anything.
    static_assert(
        std::is_convertible<PROCESS_ELEM_PAIR_LAMBDA_T,
                            std::function<void(float, float)>>::value);

    assert(!(OUTPUT_DENSE && output.active_neurons != nullptr));
    assert(!LABEL_DENSE || labels.active_neurons == nullptr);
    if (OUTPUT_DENSE && LABEL_DENSE) {
      assert(output.len == labels.len);
    }

    std::vector<bool> labels_positions_touched(labels.len);

    for (uint32_t i = 0; i < output.len; i++) {
      uint32_t active_neuron = OUTPUT_DENSE ? i : output.active_neurons[i];
      const auto found = labels.findActiveNeuron<LABEL_DENSE>(active_neuron);
      float label_val = found.activation;
      if (found.pos.has_value()) {
        labels_positions_touched[found.pos.value()] = true;
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
};

}  // namespace thirdai::bolt_v1