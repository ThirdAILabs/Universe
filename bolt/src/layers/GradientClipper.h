#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

class GradientClipper {
 public:
  virtual void clipVector(std::vector<float>& gradients) = 0;

  virtual ~GradientClipper() = default;
};

using GradientClipperPtr = std::shared_ptr<GradientClipper>;

class GradientClipperByValue : public GradientClipper {
 public:
  explicit GradientClipperByValue(float threshold) : _threshold(threshold) {}

  void clipVector(std::vector<float>& gradients) final {
    uint32_t len = gradients.size();

#pragma omp parallel for default(none) shared(len, gradients, _threshold)
    for (uint32_t i = 0; i < len; i++) {
      gradients[i] = std::clamp(gradients[i], -_threshold, _threshold);
    }
  }

 private:
  float _threshold;
};

using GradientClipperByValuePtr = std::shared_ptr<GradientClipperByValue>;

class GradientClipperByNorm : public GradientClipper {
 public:
  explicit GradientClipperByNorm(float max_norm) : _max_norm(max_norm) {}

  void clipVector(std::vector<float>& gradients) final {
    float grad_norm = compute_norm(gradients);
    uint32_t len = gradients.size();

    if (grad_norm > _max_norm) {
      float scale_factor = _max_norm / grad_norm;
#pragma omp parallel for default(none) shared(len, gradients, scale_factor)
      for (uint32_t i = 0; i < len; i++) {
        gradients[i] = gradients[i] * scale_factor;
      }
    }
  }

 private:
  static float compute_norm(const std::vector<float>& gradients) {
    float result = 0.0;

#pragma omp parallel for reduction(+ : result)
    for (float gradient : gradients) {
      result += gradient * gradient;
    }

    return std::sqrt(result);
  }
  float _max_norm;
};

using GradientClipperByNormPtr = std::shared_ptr<GradientClipperByNorm>;

class GradientClipperByFraction : public GradientClipper {
 public:
  explicit GradientClipperByFraction(float frac) : _frac(frac) {
    assert(frac >= 0 && frac <= 1);
  }

  void clipVector(std::vector<float>& gradients) final {
    float threshold = getthreshold(gradients);
    uint32_t len = gradients.size();

#pragma omp parallel for default(none) shared(len, gradients, threshold)
    for (uint32_t i = 0; i < len; i++) {
      gradients[i] = std::clamp(gradients[i], -threshold, threshold);
    }
  }

 private:
  float getthreshold(const std::vector<float>& gradients) const {
    std::vector<float> abs_gradients(gradients.size());

    #pragma omp parallel for
    for (size_t i = 0; i < gradients.size(); ++i) {
        abs_gradients[i] = std::abs(gradients[i]);
    }

    // Sort the absolute gradients in descending order
    std::sort(abs_gradients.begin(), abs_gradients.end(), std::greater<float>());

    size_t index = static_cast<size_t>(std::ceil(_frac * abs_gradients.size()));

    if (index >= abs_gradients.size()) {
        index = abs_gradients.size() - 1;
    }

    return abs_gradients[index];
}

  float _frac;
};

using GradientClipperByFractionPtr = std::shared_ptr<GradientClipperByFraction>;

}  // namespace thirdai::bolt