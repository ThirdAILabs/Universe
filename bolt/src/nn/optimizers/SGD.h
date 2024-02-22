#pragma once

#include <bolt/src/nn/optimizers/Optimizer.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <optional>

namespace thirdai::bolt {

class SGD final : public Optimizer {
 public:
  SGD(size_t rows, size_t cols, std::optional<float> grad_clip);

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

  static std::unique_ptr<SGD> fromArchive(const ar::Archive& archive);

  static std::string type() { return "sgd"; }

 private:
  inline float step(float gradient, float learning_rate) {
    if (_grad_clip) {
      return learning_rate * std::clamp(gradient, -*_grad_clip, *_grad_clip);
    }
    return learning_rate * gradient;
  }

  size_t _rows, _cols;
  std::optional<float> _grad_clip;
};

class SGDFactory final : public OptimizerFactory {
 public:
  explicit SGDFactory(std::optional<float> grad_clip = std::nullopt)
      : _grad_clip(grad_clip) {
    std::cout << "SGD Optimizer Factory called";
  }

  std::unique_ptr<Optimizer> makeOptimizer(size_t rows,
                                           size_t cols) const final {
    return std::make_unique<SGD>(rows, cols, _grad_clip);
  }

  ar::ConstArchivePtr toArchive() const final;

  static std::shared_ptr<SGDFactory> fromArchive(const ar::Archive& archive);

  static auto make() { return std::make_shared<SGDFactory>(); }

 private:
  std::optional<float> _grad_clip;
};

}  // namespace thirdai::bolt
