#include "RobeZ.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <stdexcept>

namespace thirdai::bolt {

std::string nextRobeZOpName() {
  static uint32_t constructed = 0;
  return "robez_" + std::to_string(++constructed);
}

RobeZ::RobeZ(uint64_t num_embedding_lookups, uint64_t lookup_size,
             uint64_t log_embedding_block_size, const std::string& reduction,
             std::optional<uint64_t> num_tokens_per_input,
             uint64_t update_chunk_size, uint32_t seed)
    : Op(nextRobeZOpName()) {
  EmbeddingLayerConfig config(
      /* num_embedding_lookups= */ num_embedding_lookups,
      /* lookup_size= */ lookup_size,
      /* log_embedding_block_size= */ log_embedding_block_size,
      /* update_chunk_size= */ update_chunk_size, /* reduction= */ reduction,
      /* num_tokens_per_input= */ num_tokens_per_input);

  _kernel = std::make_unique<EmbeddingLayer>(config, seed);
}

RobeZ::RobeZ(const std::string& name, const proto::bolt::RobeZ& robez_proto)
    : Op(name), _kernel(std::make_unique<EmbeddingLayer>(robez_proto)) {}

std::shared_ptr<RobeZ> RobeZ::make(uint64_t num_embedding_lookups,
                                   uint64_t lookup_size,
                                   uint64_t log_embedding_block_size,
                                   const std::string& reduction,
                                   std::optional<uint64_t> num_tokens_per_input,
                                   uint64_t update_chunk_size, uint32_t seed) {
  return std::shared_ptr<RobeZ>(new RobeZ(
      /* num_embedding_lookups= */ num_embedding_lookups,
      /* lookup_size= */ lookup_size,
      /* log_embedding_block_size= */ log_embedding_block_size,
      /* reduction= */ reduction,
      /* num_tokens_per_input= */ num_tokens_per_input,
      /* update_chunk_size= */ update_chunk_size, seed));
}

void RobeZ::forward(const ComputationList& inputs, TensorPtr& output,
                    uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 1);

  _kernel->forward(inputs[0]->tensor()->getVector(index_in_batch),
                   output->getVector(index_in_batch));
}

void RobeZ::backpropagate(ComputationList& inputs, TensorPtr& output,
                          uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  _kernel->backpropagate(inputs[0]->tensor()->getVector(index_in_batch),
                         output->getVector(index_in_batch));
}

void RobeZ::updateParameters(float learning_rate, uint32_t train_steps) {
  _kernel->updateParameters(learning_rate, train_steps, BETA1, BETA2, EPS);
}

uint32_t RobeZ::dim() const { return _kernel->getOutputDim(); }

std::optional<uint32_t> RobeZ::nonzeros(const ComputationList& inputs,
                                        bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;

  return dim();
}

void RobeZ::initOptimizer() { _kernel->initOptimizer(); }

void RobeZ::disableSparseParameterUpdates() {
  _kernel->disableSparseParameterUpdates();
}

void RobeZ::enableSparseParameterUpdates() {
  _kernel->enableSparseParameterUpdates();
}

std::vector<std::vector<float>*> RobeZ::gradients() {
  return {&_kernel->getRawEmbeddingBlockGradient()};
}

std::vector<std::vector<float>*> RobeZ::parameters() {
  return {&_kernel->getRawEmbeddingBlock()};
}

void RobeZ::summary(std::ostream& summary, const ComputationList& inputs,
                    const Computation* output) const {
  summary << "RobeZ(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name() << " [";
  _kernel->buildLayerSummary(summary);
  summary << "]";
}

void RobeZ::setSerializeOptimizer(bool should_serialize_optimizer) {
  _kernel->saveWithOptimizer(should_serialize_optimizer);
}

ComputationPtr RobeZ::apply(const ComputationList& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("RobeZ op expects a single input.");
  }

  return applyUnary(inputs.at(0));
}

ComputationPtr RobeZ::applyUnary(ComputationPtr input) {
  return Computation::make(shared_from_this(), {std::move(input)});
}

proto::bolt::Op* RobeZ::toProto(bool with_optimizer) const {
  proto::bolt::Op* op = new proto::bolt::Op();
  op->set_name(name());

  // TODO(Nicholas) move everything into this class so we don't have to deal
  // with the kernel. This will be easier to do once protobufs are added so it
  // doesn't break compatability.
  op->set_allocated_robez(_kernel->toProto(with_optimizer));

  return op;
}

SerializableParameters RobeZ::serializableParameters(
    bool with_optimizer) const {
  SerializableParameters parameters = {
      {name() + "_embedding_block", _kernel->_embedding_block.get()}};

  if (with_optimizer && _kernel->_optimizer) {
    parameters.emplace_back(name() + "_embedding_block_momentum",
                            &_kernel->_optimizer->momentum);
    parameters.emplace_back(name() + "_embedding_block_velocity",
                            &_kernel->_optimizer->velocity);
  }

  return parameters;
}

std::shared_ptr<RobeZ> RobeZ::fromProto(const std::string& name,
                                        const proto::bolt::RobeZ& robez_proto) {
  return std::shared_ptr<RobeZ>(new RobeZ(name, robez_proto));
}

template void RobeZ::save(cereal::BinaryOutputArchive&) const;

template <class Archive>
void RobeZ::save(Archive& archive) const {
  archive(cereal::base_class<Op>(this), _kernel);
}

template void RobeZ::load(cereal::BinaryInputArchive&);

template <class Archive>
void RobeZ::load(Archive& archive) {
  archive(cereal::base_class<Op>(this), _kernel);
}

std::shared_ptr<RobeZ> RobeZ::duplicateWithNewReduction(
    const std::string& reduction,
    std::optional<uint64_t> num_tokens_per_input) {
  auto new_kernel =
      _kernel->duplicateWithNewReduction(reduction, num_tokens_per_input);

  std::string new_name = nextRobeZOpName() + "_shared_" + name();
  return std::shared_ptr<RobeZ>(new RobeZ(std::move(new_kernel), new_name));
}

}  // namespace thirdai::bolt

namespace cereal {

/**
 * This is because the Op base class only uses a serialize function, whereas
 * this Op uses a load/save pair. This tells cereal to use the load save pair
 * instead of the serialize method of the parent class. See docs here:
 * https://uscilab.github.io/cereal/serialization_functions.html#inheritance
 */
template <class Archive>
struct specialize<Archive, thirdai::bolt::RobeZ,
                  cereal::specialization::member_load_save> {};

}  // namespace cereal

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::RobeZ,
                               "thirdai::bolt::nn::ops::RobeZ")