#pragma once

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/optimizers/Optimizer.h>
#include <cassert>
#include <cmath>

namespace thirdai::bolt {

class SGD final : public Optimizer {
 public:
  SGD(size_t rows, size_t cols);

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
  static inline float step(float gradient, float learning_rate) {
    return learning_rate * gradient;
  }

  size_t _rows, _cols;

  SGD() : SGD(0, 0) {}

  friend class cereal::access;

  template <class Archive>
  void save(Archive& archive) const;

  template <class Archive>
  void load(Archive& archive);
};

class SGDFactory final : public OptimizerFactory {
 public:
  std::unique_ptr<Optimizer> makeOptimizer(size_t rows,
                                           size_t cols) const final {
    return std::make_unique<SGD>(rows, cols);
  }

  static auto make() { return std::make_shared<SGDFactory>(); }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<OptimizerFactory>(this));
  }
};

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::SGDFactory)