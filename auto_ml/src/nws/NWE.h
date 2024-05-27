#pragma once

#include <cassert>
#include <memory>
#include <vector>
#include <cmath>

namespace thirdai::automl {

constexpr float pi = 3.14159265358979323846;

class Kernel {
 public:
  virtual float on(const std::vector<float>& a, const std::vector<float>& b) const = 0;
  virtual ~Kernel() {}
};

class SRPKernel final : public Kernel {
 public:
  explicit SRPKernel(uint32_t power): _power(power) {}

  float on(const std::vector<float>& a, const std::vector<float>& b) const final;

 private:
  uint32_t _power;
};

class ExponentialKernel final : public Kernel {
  // https://www.mathworks.com/help/stats/kernel-covariance-function-options.html
  // cls is characteristic length scale
 public:
  ExponentialKernel(float cls, float stdev, uint32_t power) : _cls(cls), _stdev_sq(stdev * stdev), _power(power) {}

  float on(const std::vector<float> &a, const std::vector<float> &b) const final;

 private:
  float _cls, _stdev_sq;
  uint32_t _power;
};

class NadarayaWatsonEstimator {
 public:
  explicit NadarayaWatsonEstimator(std::shared_ptr<Kernel> kernel): _kernel(std::move(kernel)) {}

  void train(std::vector<std::vector<float>> inputs, std::vector<float> outputs);

  std::vector<float> predict(const std::vector<std::vector<float>>& inputs) const;

 private:
  std::vector<std::vector<float>> _train_inputs;
  std::vector<float> _train_outputs;
  std::shared_ptr<Kernel> _kernel;
};

} // namespace thirdai::automl