#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <memory>

namespace thirdai::bolt {

class Concatenate final : public Op,
                          public std::enable_shared_from_this<Concatenate> {
 public:
  static std::shared_ptr<Concatenate> make();

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final {
    (void)learning_rate;
    (void)train_steps;
  }

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void initOptimizer() final;

  void disableSparseParameterUpdates() final {}

  void enableSparseParameterUpdates() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; };

  std::vector<std::vector<float>*> parameters() final { return {}; };

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  ComputationPtr apply(const ComputationList& inputs) final;

  proto::bolt::Op* toProto(bool with_optimizer) const final;

  SerializableParameters serializableParameters(
      bool with_optimizer) const final;

  static std::shared_ptr<Concatenate> fromProto(
      const std::string& name, const proto::bolt::Concatenate& concat_proto);

 private:
  Concatenate();

  Concatenate(const std::string& name,
              const proto::bolt::Concatenate& concat_proto);

  std::vector<uint32_t> _input_dims;
  std::vector<uint32_t> _neuron_offsets;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using ConcatenatePtr = std::shared_ptr<Concatenate>;

}  // namespace thirdai::bolt