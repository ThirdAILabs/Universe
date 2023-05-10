#include "WeightedSum.h"
#include <wrappers/src/EigenDenseWrapper.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <Eigen/src/Core/util/Constants.h>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextWeightedSumOpName() {
  static uint32_t constructed = 0;
  return "weighted_sum_" + std::to_string(++constructed);
}

WeightedSum::WeightedSum() : Op(nextWeightedSumOpName()) {}

std::shared_ptr<WeightedSum> WeightedSum::make() {
  return std::shared_ptr<WeightedSum>(new WeightedSum());
}

using EigenMatrix = Eigen::Map<
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

void WeightedSum::forward(const autograd::ComputationList& inputs,
                          tensor::TensorPtr& output, uint32_t index_in_batch,
                          bool training) {
  (void)training;
  assert(inputs.size() == 2);

  const tensor::TensorPtr& embs = inputs.at(0)->tensor();
  const tensor::TensorPtr& weights = inputs.at(1)->tensor();

  if (embs->isSparse() || weights->isSparse()) {
    throw std::invalid_argument(
        "WeightedSum op currently does not support sparse tensors.");
  }

  uint32_t seq_len = weights->dims().back();
  uint32_t emb_dim = embs->dims().back();

  EigenMatrix eigen_weights(weights->valuesAtIndex3d(index_in_batch),
                            weights->dims3d().at(1), seq_len);

  EigenMatrix eigen_embs(embs->valuesAtIndex3d(index_in_batch), seq_len,
                         emb_dim);

  EigenMatrix eigen_output(output->valuesAtIndex3d(index_in_batch),
                           weights->dims3d().at(1), emb_dim);

  eigen_output = eigen_weights * eigen_embs;
}

void WeightedSum::backpropagate(autograd::ComputationList& inputs,
                                tensor::TensorPtr& output,
                                uint32_t index_in_batch) {
  assert(inputs.size() == 2);

  const tensor::TensorPtr& embs = inputs.at(0)->tensor();
  const tensor::TensorPtr& weights = inputs.at(1)->tensor();

  uint32_t seq_len = weights->dims().back();
  uint32_t emb_dim = embs->dims().back();

  if (embs->isSparse() || weights->isSparse()) {
    throw std::invalid_argument(
        "WeightedSum op currently does not support sparse tensors.");
  }

  EigenMatrix eigen_weights(weights->valuesAtIndex3d(index_in_batch),
                            weights->dims3d().at(1), seq_len);

  EigenMatrix eigen_weights_grad(weights->gradientsAtIndex3d(index_in_batch),
                                 weights->dims3d().at(1), seq_len);

  EigenMatrix eigen_embs(embs->valuesAtIndex3d(index_in_batch), seq_len,
                         emb_dim);

  EigenMatrix eigen_embs_grad(embs->gradientsAtIndex3d(index_in_batch), seq_len,
                              emb_dim);

  EigenMatrix eigen_output(output->valuesAtIndex3d(index_in_batch),
                           weights->dims3d().at(1), emb_dim);

  EigenMatrix eigen_output_grad(output->gradientsAtIndex3d(index_in_batch),
                                weights->dims3d().at(1), emb_dim);

  eigen_embs_grad = eigen_weights.transpose() * eigen_output_grad;

  eigen_weights_grad = eigen_output_grad * eigen_embs.transpose();
}

tensor::Dims WeightedSum::dims(const autograd::ComputationList& inputs) const {
  assert(inputs.size() == 2);

  uint32_t embedding_dim = inputs.at(0)->dims().back();
  tensor::Dims output_dims = inputs.at(1)->dims();  // inputs[1] is the weights.

  output_dims.back() = embedding_dim;

  return output_dims;
}

std::optional<uint32_t> WeightedSum::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  (void)use_sparsity;
  assert(inputs.size() == 2);

  return dims(inputs).back();
}

void WeightedSum::summary(std::ostream& summary,
                          const autograd::ComputationList& inputs,
                          const autograd::Computation* output) const {
  summary << "WeightedSum(" << name()
          << "): (embeddings=" << inputs.at(0)->name()
          << ", weights=" << inputs.at(1)->name() << ") -> " << output->name();
}

autograd::ComputationPtr WeightedSum::apply(
    const autograd::ComputationPtr& embeddings,
    const autograd::ComputationPtr& weights) {
  auto emb_dims = embeddings->dims();

  if (emb_dims.size() != 2) {
    throw std::invalid_argument(
        "Expected embeddings to have 3 dimensions for WeightedSum op.");
  }

  auto weight_dims = weights->dims();

  if (weight_dims.back() != emb_dims.at(0)) {
    throw std::invalid_argument(
        "Expected the length of each set of weights to math the sequence "
        "length of the embeddings.");
  }

  return autograd::Computation::make(shared_from_this(), {embeddings, weights});
}

template void WeightedSum::serialize(cereal::BinaryInputArchive&);
template void WeightedSum::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void WeightedSum::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this));
}

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::WeightedSum)