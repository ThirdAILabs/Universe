#include "Embedding.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::ops {

std::string nextEmbeddingOpName() {
  static uint32_t constructed = 0;
  return "emb_" + std::to_string(++constructed);
}

Embedding::Embedding(uint64_t num_embedding_lookups, uint64_t lookup_size,
                     uint64_t log_embedding_block_size,
                     const std::string& reduction,
                     std::optional<uint64_t> num_tokens_per_input,
                     uint64_t update_chunk_size)
    : Op(nextEmbeddingOpName()) {
  EmbeddingLayerConfig config(
      /* num_embedding_lookups= */ num_embedding_lookups,
      /* lookup_size= */ lookup_size,
      /* log_embedding_block_size= */ log_embedding_block_size,
      /* update_chunk_size= */ update_chunk_size, /* reduction= */ reduction,
      /* num_tokens_per_input= */ num_tokens_per_input);

  _kernel = std::make_unique<EmbeddingLayer>(config);
}

std::shared_ptr<Embedding> Embedding::make(
    uint64_t num_embedding_lookups, uint64_t lookup_size,
    uint64_t log_embedding_block_size, const std::string& reduction,
    std::optional<uint64_t> num_tokens_per_input, uint64_t update_chunk_size) {
  return std::shared_ptr<Embedding>(new Embedding(
      /* num_embedding_lookups= */ num_embedding_lookups,
      /* lookup_size= */ lookup_size,
      /* log_embedding_block_size= */ log_embedding_block_size,
      /* reduction= */ reduction,
      /* num_tokens_per_input= */ num_tokens_per_input,
      /* update_chunk_size= */ update_chunk_size));
}

void Embedding::forward(const autograd::ComputationList& inputs,
                        tensor::TensorPtr& output, uint32_t index_in_batch,
                        bool training) {
  (void)training;
  assert(inputs.size() == 1);

  _kernel->forward(inputs[0]->tensor()->getVector(index_in_batch),
                   output->getVector(index_in_batch));
}

void Embedding::backpropagate(autograd::ComputationList& inputs,
                              tensor::TensorPtr& output,
                              uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  _kernel->backpropagate(inputs[0]->tensor()->getVector(index_in_batch),
                         output->getVector(index_in_batch));
}

void Embedding::updateParameters(float learning_rate, uint32_t train_steps) {
  _kernel->updateParameters(learning_rate, train_steps, BETA1, BETA2, EPS);
}

uint32_t Embedding::dim() const { return _kernel->getOutputDim(); }

std::optional<uint32_t> Embedding::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;

  return dim();
}

void Embedding::disableSparseParameterUpdates() {
  _kernel->disableSparseParameterUpdates();
}

std::vector<Op::ArrayReference> Embedding::gradients() const {
  std::vector<float>& grad = _kernel->getRawEmbeddingBlockGradient();
  return {{grad.data(), grad.size()}};
}

void Embedding::summary(std::ostream& summary,
                        const autograd::ComputationList& inputs,
                        const autograd::Computation* output) const {
  summary << "Embedding(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name() << " [";
  _kernel->buildLayerSummary(summary);
  summary << "]";
}

autograd::ComputationPtr Embedding::apply(autograd::ComputationPtr input) {
  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void Embedding::save(cereal::BinaryOutputArchive&) const;

template <class Archive>
void Embedding::save(Archive& archive) const {
  archive(cereal::base_class<Op>(this), _kernel);
}

template void Embedding::load(cereal::BinaryInputArchive&);

template <class Archive>
void Embedding::load(Archive& archive) {
  archive(cereal::base_class<Op>(this), _kernel);

  _kernel->initOptimizer();
}

}  // namespace thirdai::bolt::nn::ops

namespace cereal {

/**
 * This is because the Op base class only uses a serialize function, whereas
 * this Op uses a load/save pair. This tells cereal to use the load save pair
 * instead of the serialize method of the parent class. See docs here:
 * https://uscilab.github.io/cereal/serialization_functions.html#inheritance
 */
template <class Archive>
struct specialize<Archive, thirdai::bolt::nn::ops::Embedding,
                  cereal::specialization::member_load_save> {};

}  // namespace cereal

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::Embedding)