#include "WeightedSum.h"
#include <wrappers/src/EigenDenseWrapper.h>
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <Eigen/src/Core/Array.h>
#include <Eigen/src/Core/util/Constants.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <archive/src/ParameterReference.h>
#include <utils/Random.h>
#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

std::string nextWeightedSumOpName() {
  static uint32_t constructed = 0;
  return "weighted_sum_" + std::to_string(++constructed);
}

WeightedSum::WeightedSum(size_t n_chunks, size_t chunk_size)
    : Op(nextWeightedSumOpName()),
      _n_chunks(n_chunks),
      _chunk_size(chunk_size),
      _weights(n_chunks * chunk_size) {
  std::mt19937 rng(global_random::nextSeed());
  std::normal_distribution<float> dist(0.0, 0.01);

  auto gen = [&]() { return dist(rng); };
  std::generate(_weights.begin(), _weights.end(), gen);
}

using EigenRowMajorArray = Eigen::Map<
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using EigenVecArray =
    Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;

void WeightedSum::forward(const ComputationList& inputs, TensorPtr& output,
                          uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 1);

  const BoltVector& chunks_vec =
      inputs.at(0)->tensor()->getVector(index_in_batch);
  assert(chunks_vec.isDense());
  assert(chunks_vec.len == _n_chunks * _chunk_size);

  BoltVector& sum_vec = output->getVector(index_in_batch);
  assert(sum_vec.isDense());
  assert(sum_vec.len == _chunk_size);

  EigenRowMajorArray weights(_weights.data(), _n_chunks, _chunk_size);

  EigenVecArray sum(sum_vec.activations, _chunk_size);

  EigenRowMajorArray chunks(chunks_vec.activations, _n_chunks, _chunk_size);

  sum = (chunks.array() * weights).colwise().sum();
}

void WeightedSum::backpropagate(ComputationList& inputs, TensorPtr& output,
                                uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  BoltVector& chunks_vec = inputs.at(0)->tensor()->getVector(index_in_batch);
  assert(chunks_vec.isDense());
  assert(chunks_vec.len == _n_chunks * _chunk_size);

  const BoltVector& sum_vec = output->getVector(index_in_batch);
  assert(sum_vec.isDense());
  assert(sum_vec.len == _chunk_size);

  EigenVecArray sum_grad(sum_vec.gradients, _chunk_size);

  EigenRowMajorArray weights_grad(_optimizer->gradients.data(), _n_chunks,
                                  _chunk_size);
  EigenRowMajorArray chunks(chunks_vec.activations, _n_chunks, _chunk_size);
  weights_grad += chunks.rowwise() * sum_grad;

  if (chunks_vec.hasGradients()) {
    EigenRowMajorArray weights(_weights.data(), _n_chunks, _chunk_size);

    EigenRowMajorArray chunks_grad(chunks_vec.gradients, _n_chunks,
                                   _chunk_size);

    chunks_grad = weights.rowwise() * sum_grad;
  }
}

void WeightedSum::updateParameters(float learning_rate, uint32_t train_steps) {
  _optimizer->applyUpdate(_weights, learning_rate, train_steps);
}

uint32_t WeightedSum::dim() const { return _chunk_size; }

std::optional<uint32_t> WeightedSum::nonzeros(const ComputationList& inputs,
                                              bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;
  return _chunk_size;
}

void WeightedSum::initOptimizer() {
  _optimizer = AdamOptimizer(_weights.size());
}

void WeightedSum::disableSparseParameterUpdates() {}

void WeightedSum::enableSparseParameterUpdates() {}

std::vector<std::vector<float>*> WeightedSum::gradients() {
  return {&_optimizer->gradients};
}

std::vector<std::vector<float>*> WeightedSum::parameters() {
  return {&_weights};
}

ComputationPtr WeightedSum::applyToInputs(const ComputationList& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("QuantileMixing op expects a single input.");
  }
  return apply(inputs.at(0));
}

ar::ConstArchivePtr WeightedSum::toArchive(bool with_optimizer) const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("name", ar::str(name()));
  map->set("n_chunks", ar::u64(_n_chunks));
  map->set("chunk_size", ar::u64(_chunk_size));
  map->set("weights",
           ar::ParameterReference::make(_weights, shared_from_this()));

  if (with_optimizer && _optimizer) {
    map->set("optimizer", optimizerToArchive(*_optimizer, shared_from_this(),
                                             _n_chunks, _chunk_size));
  }

  return map;
}

std::shared_ptr<WeightedSum> WeightedSum::fromArchive(
    const ar::Archive& archive) {
  return std::shared_ptr<WeightedSum>(new WeightedSum(archive));
}

WeightedSum::WeightedSum(const ar::Archive& archive)
    : Op(archive.str("name")),
      _n_chunks(archive.u64("n_chunks")),
      _chunk_size(archive.u64("chunk_size")),
      _weights(archive.get("weights")->param().moveLoadedParameter()) {
  if (archive.contains("optimizer")) {
    _optimizer = optimizerFromArchive(*archive.get("optimizer"));
  }
}

void WeightedSum::summary(std::ostream& summary, const ComputationList& inputs,
                          const Computation* output) const {
  summary << "WeightedSum(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name() << "[n_chunks=" << _n_chunks
          << ", chunk_size=" << _chunk_size << "]";
}

void WeightedSum::setSerializeOptimizer(bool should_serialize_optimizer) {
  _should_serialize_optimizer = should_serialize_optimizer;
}

ComputationPtr WeightedSum::apply(ComputationPtr input) {
  if (input->dim() != _n_chunks * _chunk_size) {
    throw std::invalid_argument(
        "Cannot apply WeightedSum expecting " + std::to_string(_n_chunks) +
        " chunks of size " + std::to_string(_chunk_size) +
        " to input with dim " + std::to_string(input->dim()) + ".");
  }

  return Computation::make(shared_from_this(), {std::move(input)});
}

}  // namespace thirdai::bolt