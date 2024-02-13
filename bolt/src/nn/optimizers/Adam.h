#pragma once

#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/optimizers/Optimizer.h>
#include <archive/src/Archive.h>
#include <cassert>
#include <cmath>

namespace thirdai::bolt {

class Adam final : public Optimizer {
 public:
  Adam(size_t rows, size_t cols, float beta1, float beta2, float eps);

  Adam(size_t rows, size_t cols, std::vector<float> momentum,
       std::vector<float> velocity);

  explicit Adam(const ar::Archive& archive);

  void updateDense(std::vector<float>& params, std::vector<float>& grads,
                   float learning_rate, size_t train_steps) final;

  void updateSparseRows(std::vector<float>& params, std::vector<float>& grads,
                        std::vector<bool>& rows_used, float learning_rate,
                        size_t train_steps, bool reset_rows_used) final;

  void updateSparseCols(std::vector<float>& params, std::vector<float>& grads,
                        const std::vector<bool>& cols_used, float learning_rate,
                        size_t train_steps) final;

  void updateSparseRowsAndCols(std::vector<float>& params,
                               std::vector<float>& grads,
                               const std::vector<bool>& rows_used,
                               const std::vector<bool>& cols_used,
                               float learning_rate, size_t train_steps) final;

  ar::ConstArchivePtr toArchive(
      const std::shared_ptr<const Op>& op) const final;

  static std::unique_ptr<Adam> fromArchive(const ar::Archive& archive);

  static std::unique_ptr<Adam> fromOldOptimizer(AdamOptimizer&& old_opt,
                                                size_t rows, size_t cols);

  static std::string type() { return "adam"; }

 private:
  inline float momentum(float curr_momentum, float grad) const {
    return _beta1 * curr_momentum + (1 - _beta1) * grad;
  }

  inline float velocity(float curr_velocity, float grad) const {
    return _beta2 * curr_velocity + (1 - _beta2) * grad * grad;
  }

  inline float step(size_t index, float gradient, float learning_rate,
                    float b1_corrected, float b2_corrected) {
    float mom = momentum(_momentum[index], gradient);
    _momentum[index] = mom;

    float vel = velocity(_velocity[index], gradient);
    _velocity[index] = vel;

    assert(!std::isnan(gradient));
    assert(!std::isnan(mom));
    assert(!std::isnan(vel));

    return learning_rate * (mom / b1_corrected) /
           (std::sqrt(vel / b2_corrected) + _eps);
  }

  static inline float biasCorrect(float beta, uint32_t train_steps) {
    return static_cast<float>(1 - pow(beta, train_steps));
  }

  std::vector<float> _momentum;
  std::vector<float> _velocity;

  size_t _rows, _cols;

  float _beta1 = 0.9;
  float _beta2 = 0.999;
  float _eps = 1e-7;
};

class AdamFactory final : public OptimizerFactory {
 public:
  AdamFactory(float beta1, float beta2, float eps)
      : _beta1(beta1), _beta2(beta2), _eps(eps) {}

  std::unique_ptr<Optimizer> makeOptimizer(size_t rows,
                                           size_t cols) const final {
    return std::make_unique<Adam>(rows, cols, _beta1, _beta2, _eps);
  }

  ar::ConstArchivePtr toArchive() const final;

  static std::shared_ptr<AdamFactory> fromArchive(const ar::Archive& archive);

  static auto make(float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-7) {
    return std::make_shared<AdamFactory>(beta1, beta2, eps);
  }

 private:
  float _beta1 = 0.9;
  float _beta2 = 0.999;
  float _eps = 1e-7;
};

}  // namespace thirdai::bolt
