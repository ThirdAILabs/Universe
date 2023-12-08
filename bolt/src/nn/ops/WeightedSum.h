#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/ops/Op.h>
#include <memory>
#include <optional>

namespace thirdai::bolt {

class WeightedSum final : public Op,
                          public std::enable_shared_from_this<WeightedSum> {
 private:
  WeightedSum(size_t n_chunks, size_t chunk_size);

 public:
  static auto make(size_t n_chunks, size_t chunk_size) {
    return std::shared_ptr<WeightedSum>(new WeightedSum(n_chunks, chunk_size));
  }

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void initOptimizer() final;

  void disableSparseParameterUpdates() final;

  void enableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final;

  std::vector<std::vector<float>*> parameters() final;

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  void setSerializeOptimizer(bool should_serialize_optimizer) final;

  ComputationPtr apply(ComputationPtr input);

 private:
  size_t _n_chunks, _chunk_size;

  std::vector<float> _weights;

  std::optional<AdamOptimizer> _optimizer;
  bool _should_serialize_optimizer;

  WeightedSum() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Op>(this), _n_chunks, _chunk_size, _weights,
            _should_serialize_optimizer);
    if (_should_serialize_optimizer) {
      archive(_optimizer);
    }
  }
};

using WeightedSumPtr = std::shared_ptr<WeightedSum>;

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::WeightedSum)