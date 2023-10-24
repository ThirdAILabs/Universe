#include "Activation.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <proto/activation.pb.h>
#include <utils/StringManipulation.h>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

std::string nextActivationName(const std::string& name) {
  static uint32_t constructed = 0;
  return text::lower(name) + "_" + std::to_string(++constructed);
}

template <typename Impl>
Activation<Impl>::Activation() : Op(nextActivationName(Impl::name())) {}

template <typename Impl>
Activation<Impl>::Activation(const std::string& name,
                             const proto::bolt::Activation& act_proto)
    : Op(name) {
  (void)act_proto;
}

template <typename Impl>
std::shared_ptr<Activation<Impl>> Activation<Impl>::make() {
  return std::shared_ptr<Activation>(new Activation());
}

template <typename Impl>
void Activation<Impl>::forward(const ComputationList& inputs, TensorPtr& output,
                               uint32_t index_in_batch, bool training) {
  (void)training;

  const BoltVector& input_vec =
      inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& output_vec = output->getVector(index_in_batch);

  if (!input_vec.isDense()) {
    std::copy(input_vec.active_neurons,
              input_vec.active_neurons + input_vec.len,
              output_vec.active_neurons);
  }

  for (uint32_t i = 0; i < input_vec.len; i++) {
    output_vec.activations[i] = Impl::forward(input_vec.activations[i]);
  }
}

template <typename Impl>
void Activation<Impl>::backpropagate(ComputationList& inputs, TensorPtr& output,
                                     uint32_t index_in_batch) {
  BoltVector& input_vec = inputs.at(0)->tensor()->getVector(index_in_batch);
  const BoltVector& output_vec = output->getVector(index_in_batch);

  for (uint32_t i = 0; i < input_vec.len; i++) {
    float y = output_vec.activations[i];
    input_vec.gradients[i] += Impl::gradient(y) * output_vec.gradients[i];
  }
}

template <typename Impl>
void Activation<Impl>::updateParameters(float learning_rate,
                                        uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

template <typename Impl>
uint32_t Activation<Impl>::dim() const {
  return _dim;
}

template <typename Impl>
std::optional<uint32_t> Activation<Impl>::nonzeros(
    const ComputationList& inputs, bool use_sparsity) const {
  return inputs.at(0)->nonzeros(use_sparsity);
}

template <typename Impl>
void Activation<Impl>::initOptimizer() {}

template <typename Impl>
void Activation<Impl>::disableSparseParameterUpdates() {}

template <typename Impl>
void Activation<Impl>::enableSparseParameterUpdates() {}

template <typename Impl>
std::vector<std::vector<float>*> Activation<Impl>::gradients() {
  return {};
}

template <typename Impl>
std::vector<std::vector<float>*> Activation<Impl>::parameters() {
  return {};
}

template <typename Impl>
void Activation<Impl>::summary(std::ostream& summary,
                               const ComputationList& inputs,
                               const Computation* output) const {
  summary << Impl::name() << "(" << name() << "): " << inputs[0]->name()
          << " -> " << output->name();
}

template <typename Impl>
ComputationPtr Activation<Impl>::apply(const ComputationList& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("Activation op expects a single input.");
  }

  return applyUnary(inputs.at(0));
}

template <typename Impl>
ComputationPtr Activation<Impl>::applyUnary(ComputationPtr input) {
  if (dim() == 0) {
    _dim = input->dim();
  } else {
    if (dim() != input->dim()) {
      throw std::invalid_argument("Dim mismatch in tanh.");
    }
  }

  return Computation::make(this->shared_from_this(), {std::move(input)});
}

template <typename Impl>
proto::bolt::Op* Activation<Impl>::toProto(bool with_optimizer) const {
  (void)with_optimizer;

  proto::bolt::Op* op = new proto::bolt::Op();
  op->set_name(name());

  auto* activation = op->mutable_activation();
  activation->set_activation(Impl::toProto());

  return op;
}

template <typename Impl>
SerializableParameters Activation<Impl>::serializableParameters(
    bool with_optimizer) const {
  (void)with_optimizer;
  return {};
}

template <typename Impl>
std::shared_ptr<Activation<Impl>> Activation<Impl>::fromProto(
    const std::string& name, const proto::bolt::Activation& act_proto) {
  return std::shared_ptr<Activation<Impl>>(
      new Activation<Impl>(name, act_proto));
}

OpPtr activationOpFromProto(const std::string& name,
                            const proto::bolt::Activation& act_proto) {
  switch (act_proto.activation()) {
    case proto::bolt::ActivationFunction::RELU:
      return Activation<ReluImpl>::fromProto(name, act_proto);
    case proto::bolt::ActivationFunction::TANH:
      return Activation<TanhImpl>::fromProto(name, act_proto);
    default:
      throw std::invalid_argument("Activation function can not be used as op.");
  }
}

template void Activation<ReluImpl>::serialize(cereal::BinaryInputArchive&);
template void Activation<ReluImpl>::serialize(cereal::BinaryOutputArchive&);

template void Activation<TanhImpl>::serialize(cereal::BinaryInputArchive&);
template void Activation<TanhImpl>::serialize(cereal::BinaryOutputArchive&);

template <typename Impl>
template <class Archive>
void Activation<Impl>::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _dim);
}

template class Activation<ReluImpl>;
template class Activation<TanhImpl>;

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::Activation<thirdai::bolt::ReluImpl>)
CEREAL_REGISTER_TYPE_WITH_NAME(
    thirdai::bolt::Activation<thirdai::bolt::TanhImpl>,
    "thirdai::bolt::nn::ops::Tanh")