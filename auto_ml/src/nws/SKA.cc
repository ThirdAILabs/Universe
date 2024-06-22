#include "SKA.h"

namespace thirdai::automl {

float Theta::between(const std::vector<float>& a,
                     const std::vector<float>& b) const {
  assert(a.size() == b.size());
  float dot = 0, l2_a = 0, l2_b = 0;
  for (size_t i = 0; i < a.size(); i++) {
    dot += a[i] * b[i];
    l2_a += a[i] * a[i];
    l2_b += b[i] * b[i];
  }
  l2_a = std::sqrt(l2_a);
  l2_b = std::sqrt(l2_b);
  // Avoid rounding errors.
  const float arg =
      std::min<float>(std::max<float>(dot / (l2_a * l2_b), -1.0), 1.0);
  return std::acos(arg);
}

float L2Distance::between(const std::vector<float>& a,
                          const std::vector<float>& b) const {
  assert(a.size() == b.size());
  float sum_of_squares = 0;
  for (size_t i = 0; i < a.size(); i++) {
    const float diff = a[i] - b[i];
    sum_of_squares += diff * diff;
  }
  return std::sqrt(sum_of_squares);
}

void SKASampler::use(uint32_t k) {
  assert(k >= _used_train_inputs.size());
  assert(k > 0);
  assert(k <= (_used_train_inputs.size() + _train_inputs.size()));
  if (_used_train_inputs.empty()) {
    // Assume the dataset is shuffled anyway.
    useSample(/* sample_idx= */ 0);
  }
  while (_used_train_inputs.size() < k) {
    uint32_t argmax = (*_train_inputs.begin()).first;
    float max_min_distance = 0;
    for (const auto& [sample_idx, unused_input] : _train_inputs) {
      float min_distance = std::numeric_limits<float>::max();
      for (const auto& used_input : _used_train_inputs) {
        min_distance = std::min(_distance->between(used_input, unused_input),
                                min_distance);
      }
      if (min_distance > max_min_distance) {
        argmax = sample_idx;
        max_min_distance = min_distance;
      }
    }
    useSample(argmax);
  }
}

void SKASampler::useSample(uint32_t sample_idx) {
  _used_train_inputs.push_back(std::move(_train_inputs[sample_idx]));
  _used_train_outputs.push_back(_train_outputs[sample_idx]);
  _train_inputs.erase(sample_idx);
  _train_outputs.erase(sample_idx);
}

std::vector<float> SparseKernelApproximation::predict(const std::vector<std::vector<float>> &inputs) const {
  std::vector<float> outputs(inputs.size());
#pragma omp parallel for default(none) shared(inputs, outputs)
  for (size_t i = 0; i < inputs.size(); i++) {
    float num = 0, denom = 0;
    for (size_t j = 0; j < _train_inputs.size(); j++) {
      const float sim = _kernel->on(inputs[i], _train_inputs[j]);
      num += _alphas[j] * sim * _train_outputs[j];
      denom += _alphas[j] * sim;
    }
    outputs[i] = denom ? num / denom : 0;
  }
  return outputs;
}

std::vector<std::vector<float>> kMatrix(const std::shared_ptr<Kernel> &kernel, const std::vector<std::vector<float>>& vectors) {   
  std::vector<std::vector<float>> output(vectors.size(), std::vector<float>(vectors.size()));
#pragma omp parallel for default(none) shared(kernel, vectors, output)
  for (uint32_t i = 0; i < vectors.size(); i++) {
    for (uint32_t j = 0; j <= i; j++) {
      output[i][j] = kernel->on(vectors[i], vectors[j]);
    }
  }
#pragma omp parallel for default(none) shared(kernel, vectors, output)
  for (uint32_t j = 0; j < vectors.size(); j++) {
    for (uint32_t i = 0; i < j; i++) {
      output[i][j] = output[j][i];
    }
  }
  return output;
}

}  // namespace thirdai::automl