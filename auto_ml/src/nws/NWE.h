#pragma once

#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

namespace thirdai::automl {

constexpr float pi = 3.14159265358979323846;

class Kernel {
 public:
  virtual float on(const std::vector<float>& a,
                   const std::vector<float>& b) const = 0;
  virtual ~Kernel() {}
};

class SRPKernel final : public Kernel {
 public:
  explicit SRPKernel(uint32_t power) : _power(power) {}

  float on(const std::vector<float>& a,
           const std::vector<float>& b) const final;

 private:
  uint32_t _power;
};

class L2Kernel final : public Kernel {
 public:
  L2Kernel(float bandwidth, uint32_t power)
      : _bandwidth(bandwidth), _power(power) {}

  float on(const std::vector<float>& a,
           const std::vector<float>& b) const final;

 private:
  float _bandwidth;
  uint32_t _power;
};

class NadarayaWatsonEstimator {
 public:
  explicit NadarayaWatsonEstimator(std::shared_ptr<Kernel> kernel)
      : _kernel(std::move(kernel)) {}

  void train(std::vector<std::vector<float>> inputs,
             std::vector<float> outputs);

  std::vector<float> predict(
      const std::vector<std::vector<float>>& inputs) const;

 private:
  std::vector<std::vector<float>> _train_inputs;
  std::vector<float> _train_outputs;
  std::shared_ptr<Kernel> _kernel;
};

}  // namespace thirdai::automl