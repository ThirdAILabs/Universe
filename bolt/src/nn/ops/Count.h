#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <limits>
#include <memory>
#include <vector>

namespace thirdai::bolt::nn::ops {

/**
 * Counts the number of elements in the input vector.
 */
class Count final : public Op, public std::enable_shared_from_this<Count> {
 public:
  static std::shared_ptr<Count> make();

  /**
   * `inputs` will always have size=1.
   */
  void forward(const autograd::ComputationList& inputs,
               tensor::TensorPtr& output, uint32_t index_in_batch,
               bool training) final;

  /**
   * This op doesn't perform computations so the backpropagate and
   * updateParameters methods are no-ops.
   */
  void backpropagate(autograd::ComputationList& inputs,
                     tensor::TensorPtr& output, uint32_t index_in_batch) final {
    (void)inputs;
    (void)output;
    (void)index_in_batch;
  }

  void updateParameters(float learning_rate, uint32_t train_steps) final {
    (void)learning_rate;
    (void)train_steps;
  }

  uint32_t dim() const final { return std::numeric_limits<uint32_t>::max(); }

  std::optional<uint32_t> nonzeros(const autograd::ComputationList& inputs,
                                   bool use_sparsity) const final {
    (void)inputs;
    (void)use_sparsity;
    return 1;
  }

  // No-op
  void disableSparseParameterUpdates() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; }

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

  void setSerializeOptimizer(bool should_serialize_optimizer) final {
    (void)should_serialize_optimizer;
  }

  /**
   * Applies the op to an input tensor and yields a new output tensor. Used to
   * add the op to a computation graph.
   */
  autograd::ComputationPtr apply(autograd::ComputationPtr input);

  static auto cast(const ops::OpPtr& op) {
    return std::dynamic_pointer_cast<Count>(op);
  }

 private:
  Count();

  friend class cereal::access;

  // We use save/load instead of serialize so we can ensure the optimizer is
  // initialized when the model is loaded.
  template <class Archive>
  void save(Archive& archive) const;

  template <class Archive>
  void load(Archive& archive);
};

}  // namespace thirdai::bolt::nn::ops