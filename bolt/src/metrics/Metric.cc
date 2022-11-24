#include "Metric.h"
namespace thirdai::bolt {

void CategoricalCrossEntropy::record(const BoltVector& outputs,
                                     const BoltVector& labels) {
  float sample_loss = 0;
  const float EPS = 1e-7;
  if (outputs.isDense()) {
    if (labels.isDense()) {
      // (Dense Output, Dense Labels)
      // If both are dense, they're expected to have the same length.
      // In this case, we may simply run over the dense vectors and compute
      // sum((p_i)log(q_i)).
      assert(outputs.len == labels.len);
      for (uint32_t i = 0; i < outputs.len; i++) {
        sample_loss +=
            labels.activations[i] * std::log(outputs.activations[i] + EPS);
      }
    } else {
      // (Dense Output, Sparse Labels)
      for (uint32_t i = 0; i < outputs.len; i++) {
        const uint32_t* label_start = labels.active_neurons;
        const uint32_t* label_end = labels.active_neurons + labels.len;

        // Find the position of the active neuron if it exists in the labels.
        const uint32_t* label_query = std::find(label_start, label_end, i);

        if (label_query != label_end) {
          // In this case, we have found the labels. Other label activations
          // are 0, so we can ignore (0*log(whatever)).
          //
          // Compute label_index to lookup the value from labels
          // sparse-vector.
          size_t label_index = std::distance(label_start, label_query);

          sample_loss += labels.activations[label_index] *
                         std::log(outputs.activations[i] + EPS);
        }
      }
    }
  } else {
    // outputs is sparse.
    if (labels.isDense()) {
      // (Sparse Output, Dense Label)
      for (uint32_t i = 0; i < labels.len; i++) {
        const uint32_t* output_start = outputs.active_neurons;
        const uint32_t* output_end = outputs.active_neurons + outputs.len;

        // Find the position of the active neuron if it exists in the labels.
        const uint32_t* output_query = std::find(output_start, output_end, i);

        if (output_query != output_end) {
          // Compute output_index to lookup the value from output
          // sparse-vector.
          size_t output_index = std::distance(output_start, output_query);

          sample_loss += labels.activations[i] *
                         std::log(outputs.activations[output_index] + EPS);
        } else {
          float output_activation = 0.0F;
          sample_loss +=
              labels.activations[i] * std::log(output_activation + EPS);
        }
      }
    } else {
      for (uint32_t i = 0; i < outputs.len; i++) {
        const uint32_t* label_start = labels.active_neurons;
        const uint32_t* label_end = labels.active_neurons + labels.len;

        // Find the position of the active neuron if it exists in the labels.
        const uint32_t* label_query =
            std::find(label_start, label_end, outputs.active_neurons[i]);
        if (label_query != label_end) {
          // In this case, we have found the labels. Other label activations
          // are 0, so we can ignore (0*log(whatever)).
          //
          // Compute label_index to lookup the value from labels
          // sparse-vector.
          size_t label_index = std::distance(label_start, label_query);

          sample_loss += labels.activations[label_index] *
                         std::log(outputs.activations[i] + EPS);
        }
      }
    }
  }

  MetricUtilities::incrementAtomicFloat(_sum, -1 * sample_loss);
  _num_samples++;
}
}  // namespace thirdai::bolt
