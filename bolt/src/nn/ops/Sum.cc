#include "Sum.h"
#include <wrappers/src/EigenDenseWrapper.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextSumOpName() {
  static uint32_t constructed = 0;
  return "sum_" + std::to_string(++constructed);
}

Sum::Sum() : Op(nextSumOpName()) {}

std::shared_ptr<Sum> Sum::make() { return std::shared_ptr<Sum>(new Sum()); }

using EigenArray = Eigen::Map<Eigen::ArrayXf>;

void Sum::forward(const autograd::ComputationList& inputs,
                  tensor::TensorPtr& output, uint32_t index_in_batch,
                  bool training) {
  (void)training;
  assert(inputs.size() == 2);

  const auto& lhs = inputs.at(0)->tensor();
  const auto& rhs = inputs.at(1)->tensor();

  if (lhs->isSparse() || rhs->isSparse()) {
    throw std::invalid_argument(
        "Sum op currently does not support sparse tensors.");
  }

  uint32_t len = lhs->dims3d().at(1) * lhs->dims3d().at(2);

  EigenArray lhs_eigen(lhs->valuesAtIndex3d(index_in_batch), len);
  EigenArray rhs_eigen(rhs->valuesAtIndex3d(index_in_batch), len);
  EigenArray output_eigen(output->valuesAtIndex3d(index_in_batch), len);

  output_eigen = lhs_eigen + rhs_eigen;
}

void Sum::backpropagate(autograd::ComputationList& inputs,
                        tensor::TensorPtr& output, uint32_t index_in_batch) {
  assert(inputs.size() == 2);

  const auto& lhs = inputs.at(0)->tensor();
  const auto& rhs = inputs.at(1)->tensor();

  if (lhs->isSparse() || rhs->isSparse()) {
    throw std::invalid_argument(
        "Sum op currently does not support sparse tensors.");
  }

  uint32_t len = lhs->dims3d().at(1) * lhs->dims3d().at(2);

  EigenArray lhs_grad_eigen(lhs->gradientsAtIndex3d(index_in_batch), len);
  EigenArray rhs_grad_eigen(rhs->gradientsAtIndex3d(index_in_batch), len);
  EigenArray output_grad_eigen(output->gradientsAtIndex3d(index_in_batch), len);

  lhs_grad_eigen += output_grad_eigen;
  rhs_grad_eigen += output_grad_eigen;
}

tensor::Dims Sum::dims(const autograd::ComputationList& inputs) const {
  assert(inputs.size() == 2);

  return inputs.at(0)->dims();
}

std::optional<uint32_t> Sum::nonzeros(const autograd::ComputationList& inputs,
                                      bool use_sparsity) const {
  (void)use_sparsity;

  return dims(inputs).back();
}

void Sum::summary(std::ostream& summary,
                  const autograd::ComputationList& inputs,
                  const autograd::Computation* output) const {
  summary << "Sum(" << name() << "): (" << inputs.at(0)->name() << ", "
          << inputs.at(1)->name() << ") -> " << output->name();
}

autograd::ComputationPtr Sum::apply(const autograd::ComputationPtr& lhs,
                                    const autograd::ComputationPtr& rhs) {
  if (!tensor::areDimsEq(lhs->dims(), rhs->dims())) {
    throw std::invalid_argument(
        "Can only apply sum to tensors with the same shape. Received tensors "
        "with shape " +
        tensor::toString(lhs->dims()) + " and " +
        tensor::toString(rhs->dims()) + ".");
  }

  return autograd::Computation::make(shared_from_this(), {lhs, rhs});
}

template void Sum::serialize(cereal::BinaryInputArchive&);
template void Sum::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Sum::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this));
}

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::Sum)