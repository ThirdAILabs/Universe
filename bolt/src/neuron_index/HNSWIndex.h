#pragma once

#include <bolt/src/neuron_index/NeuronIndex.h>
#include <hashing/src/HashUtils.h>
#include <search/src/HNSW.h>
#include <utils/Random.h>
#include <random>
#include <stdexcept>

namespace thirdai::bolt::nn {

class HNSWIndex final : public NeuronIndex {
 public:
  HNSWIndex(size_t max_nbrs, size_t construction_buf_size,
            size_t search_buf_size)
      : _max_nbrs(max_nbrs),
        _construction_buffer_size(construction_buf_size),
        _search_buffer_size(search_buf_size) {}

  void query(const BoltVector& input, BoltVector& output,
             const BoltVector* labels) const final;

  void buildIndex(const std::vector<float>& weights, uint32_t dim,
                  bool use_new_seed) final;

  void autotuneForNewSparsity(uint32_t dim, uint32_t prev_dim, float sparsity,
                              bool experimental_autotune) final {
    (void)dim;
    (void)prev_dim;
    (void)sparsity;
    (void)experimental_autotune;
  }

  void summarize(std::ostream& summary) const final {
    summary << "HNSW Index("
            << "M=" << _max_nbrs
            << " construction_buf_size=" << _construction_buffer_size
            << " search_buf_size=" << _search_buffer_size << ")";
  }

  auto avgVisited() const { return _index->avgVisited(); }

 private:
  size_t _max_nbrs;
  size_t _construction_buffer_size;
  size_t _search_buffer_size;

  std::unique_ptr<search::hnsw::HNSW> _index;

  std::vector<uint32_t> _rand_neurons;
};

}  // namespace thirdai::bolt::nn