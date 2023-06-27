#pragma once

#include <cereal/access.hpp>
#include <vector>

namespace thirdai::bolt::nn::optimizers {

class Optimizer {
 public:
  virtual void updateDense(std::vector<float>& params,
                           std::vector<float>& grads, float learning_rate,
                           size_t train_steps) = 0;

  virtual void updateSparseRows(std::vector<float>& params,
                                std::vector<float>& grads,
                                const std::vector<bool>& rows_used,
                                float learning_rate, size_t train_steps) = 0;

  virtual void updateSparseCols(std::vector<float>& params,
                                std::vector<float>& grads,
                                const std::vector<bool>& cols_used,
                                float learning_rate, size_t train_steps) = 0;

  virtual void updateSparseRowsAndCols(std::vector<float>& params,
                                       std::vector<float>& grads,
                                       const std::vector<bool>& rows_used,
                                       const std::vector<bool>& cols_used,
                                       float learning_rate,
                                       size_t train_steps) = 0;

  void setSerializeState(bool serialize_state) {
    _serialize_state = serialize_state;
  }

 protected:
  bool shouldSerializeState() const { return _serialize_state; }

 private:
  bool _serialize_state = false;

  friend class cereal::access;

  template <class Archive>
  void save(Archive& archive) const {
    (void)archive;
  }

  template <class Archive>
  void load(Archive& archive) {
    (void)archive;
  }
};

using OptimizerPtr = std::unique_ptr<Optimizer>;

class Factory {
 public:
  virtual OptimizerPtr makeOptimizer(size_t rows, size_t cols) const = 0;
};

}  // namespace thirdai::bolt::nn::optimizers