#include "Sum.h"
#include <wrappers/src/EigenDenseWrapper.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

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

  uint32_t len = lhs->innerDim3d() * lhs->dims().back();
  uint32_t offset = index_in_batch * lhs->innerDim3d();

  EigenArray lhs_eigen(lhs->activations().data() + offset, len);
  EigenArray rhs_eigen(rhs->activations().data() + offset, len);
  EigenArray output_eigen(output->activations().data() + offset, len);

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

  uint32_t len = lhs->innerDim3d() * lhs->dims().back();
  uint32_t offset = index_in_batch * lhs->innerDim3d();

  EigenArray lhs_grad_eigen(lhs->gradients().data() + offset, len);
  EigenArray rhs_grad_eigen(rhs->gradients().data() + offset, len);
  EigenArray output_grad_eigen(output->gradients().data() + offset, len);

  lhs_grad_eigen += output_grad_eigen;
  rhs_grad_eigen += output_grad_eigen;
}

tensor::Dims Sum::dims(const autograd::ComputationList& inputs) const {
  assert(inputs.size() == 2);

  return inputs.at(0)->dims();
}

std::optional<uint32_t> Sum::nonzeros(const autograd::ComputationList& inputs,
                                      bool use_sparsity) const {
  assert(inputs.size() == 2);

  return inputs.at(0)->nonzeros(use_sparsity);
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

}  // namespace thirdai::bolt::nn::ops