#pragma once

#include <cereal/access.hpp>
#include <vector>

namespace thirdai::bolt::nn::optimizers {

class Optimizer {
 public:
  virtual void updateDense(std::vector<float>& params,
                           std::vector<float>& grads, float learning_rate,
                           size_t train_steps) = 0;

  /**
   * In this method rows_used is not a const reference because we want the
   * option to reset rows_used back to false as it is iterated over. This
   * has a performance impact for embedding layers. For other methods are used
   * in fully connected layers and where the layer resets which rows/cols are
   * used after updating the bias parameters.
   */
  virtual void updateSparseRows(std::vector<float>& params,
                                std::vector<float>& grads,
                                std::vector<bool>& rows_used,
                                float learning_rate, size_t train_steps,
                                bool reset_rows_used) = 0;

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

  virtual ~Optimizer() = default;

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

  virtual ~Factory() = default;
};

}  // namespace thirdai::bolt::nn::optimizers