#pragma once

#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/ops/Op.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

class Embedding final : public Op,
                        public std::enable_shared_from_this<Embedding> {
 private:
  Embedding(size_t dim, size_t input_dim, const std::string& activation,
            bool bias);

  explicit Embedding(const ar::Archive& archive);

 public:
  static auto make(size_t dim, size_t input_dim, const std::string& activation,
                   bool bias = true) {
    return std::shared_ptr<Embedding>(
        new Embedding(dim, input_dim, activation, bias));
  }

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void forward(const BoltVector& tokens, float* output) const;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  void initOptimizer(const OptimizerFactoryPtr& optimizer_factory,
                     bool replace_existing_optimizer) final;

  uint32_t dim() const final { return _dim; }

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final {
    (void)inputs;
    (void)use_sparsity;
    return dim();
  }

  void disableSparseParameterUpdates() final {
    _disable_sparse_parameter_updates = true;
  }

  void enableSparseParameterUpdates() final {
    _disable_sparse_parameter_updates = false;
  }

  std::vector<std::vector<float>*> gradients() final {
    return {&_embedding_gradients, &_bias_gradients};
  }

  std::vector<std::vector<float>*> parameters() final {
    return {&_embeddings, &_biases};
  }

  ComputationPtr applyToInputs(const ComputationList& inputs) final;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::shared_ptr<Embedding> fromArchive(const ar::Archive& archive);

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  void setSerializeOptimizer(bool should_serialize_optimizer) final {
    _should_serialize_optimizer = should_serialize_optimizer;
  }

  std::vector<std::pair<std::string, double>> parameterAndGradNorms()
      const final;

  ComputationPtr apply(ComputationPtr input);

  uint32_t inputDim() const { return _input_dim; }

  ActivationFunction activation() const { return _act_func; }

  bool useBias() const { return _bias; }

  const float* embeddingsPtr() const { return _embeddings.data(); }

  const float* biasesPtr() const { return _biases.data(); }

  void setEmbeddings(const float* embeddings) {
    std::copy(embeddings, embeddings + _dim * _input_dim, _embeddings.begin());
  }

  void setBiases(const float* biases) {
    std::copy(biases, biases + _dim, _biases.begin());
  }

  static auto cast(const OpPtr& op) {
    return std::dynamic_pointer_cast<Embedding>(op);
  }

  static std::string type() { return "emb"; }

  void swapActivation(ActivationFunction new_act) { _act_func = new_act; }

  void useTorchInitialization() final;

 private:
  inline void applyActivationFunction(float* activations) const;

  void applyActivationFunctionGrad(const float* activations, float* gradients);

  inline const float* embedding(size_t token) const {
    return _embeddings.data() + token * _dim;
  }

  inline float* gradients(size_t token) {
    return _embedding_gradients.data() + token * _dim;
  }

  size_t _dim, _input_dim;
  bool _bias;
  ActivationFunction _act_func;

  std::vector<float> _embeddings;
  std::vector<float> _biases;

  bool _disable_sparse_parameter_updates;

  std::vector<float> _embedding_gradients;
  std::vector<float> _bias_gradients;
  bool _should_serialize_optimizer;

  OptimizerPtr _embedding_optimizer;
  OptimizerPtr _bias_optimizer;
  std::vector<bool> _embeddings_used;

  Embedding() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using EmbeddingPtr = std::shared_ptr<Embedding>;

}  // namespace thirdai::bolt