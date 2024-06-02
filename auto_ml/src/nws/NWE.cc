#include "NWE.h"
#include <cmath>
#include <iostream>

namespace thirdai::automl {

float SRPKernel::on(const std::vector<float>& a,
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
  const float theta = std::acos(arg);
  return std::pow(1 - (theta / pi), _power);
}

float ExponentialKernel::on(const std::vector<float>& a,
                            const std::vector<float>& b) const {
  assert(a.size() == b.size());
  float r = 0;
  for (size_t i = 0; i < a.size(); i++) {
    const float diff = a[i] - b[i];
    r += diff * diff;
  }
  r = std::sqrt(r);
  const float kernel = _stdev_sq * std::exp(-r / _cls);
  return std::pow(kernel, _power);
}

void NadarayaWatsonEstimator::train(std::vector<std::vector<float>> inputs,
                                    std::vector<float> outputs) {
  assert(inputs.size() == outputs.size());
  _train_inputs = std::move(inputs);
  _train_outputs = std::move(outputs);
}

std::vector<float> NadarayaWatsonEstimator::predict(
    const std::vector<std::vector<float>>& inputs) const {
  std::vector<float> outputs(inputs.size());
#pragma omp parallel for default(none) shared(inputs, outputs, std::cout)
  for (size_t i = 0; i < inputs.size(); i++) {
    float num = 0, denom = 0;
    for (size_t j = 0; j < _train_inputs.size(); j++) {
      const float sim = _kernel->on(inputs[i], _train_inputs[j]);
      num += sim * _train_outputs[j];
      denom += sim;
    }
    outputs[i] = denom ? num / denom : 0;
  }
  return outputs;
}

}  // namespace thirdai::automl