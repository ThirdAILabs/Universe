#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/optimizers/Adam.h>
#include <memory>
#include <optional>
#include <type_traits>

namespace thirdai::bolt {

class WeightedSum final : public Op,
                          public std::enable_shared_from_this<WeightedSum> {
 private:
  WeightedSum(size_t n_chunks, size_t chunk_size);

  explicit WeightedSum(const ar::Archive& archive);

 public:
  static auto make(size_t n_chunks, size_t chunk_size) {
    return std::shared_ptr<WeightedSum>(new WeightedSum(n_chunks, chunk_size));
  }

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  void initOptimizer(const OptimizerFactoryPtr& optimizer_factory) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final;

  void enableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final;

  std::vector<std::vector<float>*> parameters() final;

  ComputationPtr applyToInputs(const ComputationList& inputs) final;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::shared_ptr<WeightedSum> fromArchive(const ar::Archive& archive);

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  void setSerializeOptimizer(bool should_serialize_optimizer) final;

  std::vector<std::pair<std::string, double>> parameterAndGradNorms()
      const final;

  ComputationPtr apply(ComputationPtr input);

  static std::string type() { return "weighted_sum"; }

 private:
  size_t _n_chunks, _chunk_size;

  std::vector<float> _weights;
  std::vector<float> _gradients;

  OptimizerPtr _optimizer;
  bool _should_serialize_optimizer;

  WeightedSum() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Op>(this), _n_chunks, _chunk_size, _weights,
            _should_serialize_optimizer);

    if (_should_serialize_optimizer &&
        std::is_same_v<Archive, cereal::BinaryInputArchive>) {
      AdamOptimizer optimizer;

      archive(optimizer);

      _optimizer =
          Adam::fromOldOptimizer(std::move(optimizer), 1, _weights.size());

      _gradients.assign(_weights.size(), 0.0);
    }
  }
};

using WeightedSumPtr = std::shared_ptr<WeightedSum>;

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::WeightedSum)