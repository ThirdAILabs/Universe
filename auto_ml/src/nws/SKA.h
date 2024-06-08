#pragma once

#include <cassert>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace thirdai::automl {

class Distance {
 public:
  virtual float between(const std::vector<float>& a,
                        const std::vector<float>& b) const = 0;
  virtual ~Distance() {}
};

class Theta final : public Distance {
 public:
  Theta() {}
  float between(const std::vector<float>& a,
                const std::vector<float>& b) const final;
};

class L2Distance final : public Distance {
 public:
  L2Distance() {}
  float between(const std::vector<float>& a,
                const std::vector<float>& b) const final;
};

class SparseKernelApproximation {
 public:
  SparseKernelApproximation(std::shared_ptr<Distance> distance,
                            std::vector<std::vector<float>> train_inputs,
                            std::vector<float> train_outputs)
      : _distance(std::move(distance)),
        _train_inputs(toMap(std::move(train_inputs))),
        _train_outputs(toMap(std::move(train_outputs))) {
    assert(_train_inputs.size() == _train_outputs.size());
  }

  void use(uint32_t k);

  std::pair<std::vector<std::vector<float>>, std::vector<float>> usedSamples() {
    return {_used_train_inputs, _used_train_outputs};
  }

 private:
  void useSample(uint32_t sample_idx);

  template <typename V>
  static std::unordered_map<uint32_t, V> toMap(std::vector<V>&& vec) {
    std::unordered_map<uint32_t, V> map;
    for (uint32_t i = 0; i < vec.size(); i++) {
      map[i] = std::move(vec[i]);
    }
    return map;
  }

  std::shared_ptr<Distance> _distance;
  std::unordered_map<uint32_t, std::vector<float>> _train_inputs;
  std::unordered_map<uint32_t, float> _train_outputs;
  std::vector<std::vector<float>> _used_train_inputs;
  std::vector<float> _used_train_outputs;
};

}  // namespace thirdai::automl