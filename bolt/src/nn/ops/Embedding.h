#pragma once

#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/ops/Op.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

class Embedding final : public Op,
                        public std::enable_shared_from_this<Embedding> {
 private:
  Embedding(size_t dim, size_t input_dim, const std::string& activation,
            bool bias);

 public:
  static auto make(size_t dim, size_t input_dim, const std::string& activation,
                   bool bias = true) {
    return std::shared_ptr<Embedding>(
        new Embedding(dim, input_dim, activation, bias));
  }

  void forward(const autograd::ComputationList& inputs,
               tensor::TensorPtr& output, uint32_t index_in_batch,
               bool training) final;

  void backpropagate(autograd::ComputationList& inputs,
                     tensor::TensorPtr& output, uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t dim() const final { return _dim; }

  std::optional<uint32_t> nonzeros(const autograd::ComputationList& inputs,
                                   bool use_sparsity) const final {
    (void)inputs;
    (void)use_sparsity;
    return dim();
  }

  void disableSparseParameterUpdates() final {
    _disable_sparse_parameter_updates = true;
  }

  std::vector<std::vector<float>*> gradients() final {
    return {&_embedding_optimizer->gradients, &_bias_optimizer->gradients};
  }

  std::vector<std::vector<float>*> parameters() final {
    return {&_embeddings, &_biases};
  }

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

  void setSerializeOptimizer(bool should_serialize_optimizer) final {
    _should_serialize_optimizer = should_serialize_optimizer;
  }

  autograd::ComputationPtr apply(autograd::ComputationPtr input);

 private:
  void sparseEmbeddingUpdate(float learning_rate, uint32_t train_steps);

  size_t _dim, _input_dim;
  bool _bias;
  ActivationFunction _act_func;

  std::vector<float> _embeddings;
  std::vector<float> _biases;

  bool _disable_sparse_parameter_updates;
  bool _should_serialize_optimizer;

  std::optional<AdamOptimizer> _embedding_optimizer = std::nullopt;
  std::optional<AdamOptimizer> _bias_optimizer = std::nullopt;
  std::vector<bool> _embeddings_used;

  Embedding() {}

  friend class cereal::access;

  template <class Archive>
  void save(Archive& archive) const;

  template <class Archive>
  void load(Archive& archive);
};

}  // namespace thirdai::bolt::nn::ops