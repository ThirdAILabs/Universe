#include "HNSWIndex.h"

namespace thirdai::bolt::nn {

void HNSWIndex::query(const BoltVector& input, BoltVector& output,
                      const BoltVector* labels) const {
  (void)labels;

  if (!input.isDense()) {
    throw std::invalid_argument(
        "HNSW index can only be used for dense inputs.");
  }

  size_t buffer_size = std::max<size_t>(output.len, _search_buffer_size);

  auto topk = _index->querySet(input.activations, output.len, buffer_size);

  if (topk.size() < output.len) {
    uint32_t rand_offset =
        hashing::simpleIntegerHash(*topk.begin()) % _rand_neurons.size();

    while (topk.size() < output.len) {
      topk.insert(_rand_neurons[rand_offset]);
      rand_offset++;
      if (rand_offset == _rand_neurons.size()) {
        rand_offset = 0;
      }
    }
  }

  std::copy(topk.begin(), topk.end(), output.active_neurons);
}

void HNSWIndex::buildIndex(const std::vector<float>& weights, uint32_t dim,
                           bool use_new_seed) {
  (void)use_new_seed;

  if (weights.size() % dim != 0) {
    throw std::invalid_argument(
        "Length of weights must be a multiple of the dim.");
  }

  size_t prev_dim = weights.size() / dim;

  _index = std::make_unique<search::hnsw::HNSW>(
      _max_nbrs, prev_dim, dim, weights.data(), _construction_buffer_size);

  _rand_neurons.assign(dim, 0);
  std::mt19937 rng(global_random::nextSeed());
  std::iota(_rand_neurons.begin(), _rand_neurons.end(), 0);
  std::shuffle(_rand_neurons.begin(), _rand_neurons.end(), rng);
}

}  // namespace thirdai::bolt::nn