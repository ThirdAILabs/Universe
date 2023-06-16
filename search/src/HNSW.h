#pragma once

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace thirdai::search::hnsw {

struct NodeDistPair {
  NodeDistPair(uint32_t n, float d) : node(n), dist(d) {}

  uint32_t node;
  float dist;
};

struct PairGreater {
  constexpr bool operator()(const NodeDistPair& a,
                            const NodeDistPair& b) const {
    return a.dist > b.dist;
  }
};

struct PairLess {
  constexpr bool operator()(const NodeDistPair& a,
                            const NodeDistPair& b) const {
    return a.dist < b.dist;
  }
};

// Use PairGreater if smallest is closest.
using ClosestQueue =
    std::priority_queue<NodeDistPair, std::vector<NodeDistPair>, PairGreater>;

// Use PairLess if largest is furthest.
using FurthestQueue =
    std::priority_queue<NodeDistPair, std::vector<NodeDistPair>, PairLess>;

class HNSW {
 public:
  HNSW(size_t max_nbrs, size_t dim, size_t n_nodes, const float* data,
       size_t construction_buf_size, size_t num_initializations = 100);

  std::vector<uint32_t> query(const float* query, uint32_t k,
                              size_t search_buffer_size,
                              size_t num_initializations = 100);

  std::unordered_set<uint32_t> querySet(const float* query, uint32_t k,
                                        size_t search_buffer_size,
                                        size_t num_initializations = 100);

  double avgVisited() const {
    return static_cast<double>(_visited_count.load()) / _n_queries.load();
  }

 private:
  void insert(const float* data, size_t search_buffer_size,
              size_t num_initializations);

  uint32_t searchInitialization(const float* query,
                                size_t num_initializations) const;

  std::pair<ClosestQueue, size_t> beamSearch(const float* query,
                                             uint32_t entry_node,
                                             size_t buffer_size);

  ClosestQueue selectNeighbors(ClosestQueue& candidates) const;

  void connectNeighbors(ClosestQueue& neighbors, uint32_t new_node);

  float distance(const float* a, const float* b) const {
    return l2Distance(a, b, _dim);
    // return cosineDistance(a, b, _dim);
    // return innerProductDistance(a, b, _dim);
  }

  static inline float cosineDistance(const float* a, const float* b,
                                     size_t dim) {
    float dot_product = 0.0;
    float a_mag = 0.0;
    float b_mag = 0.0;

    for (size_t i = 0; i < dim; i++) {
      dot_product += a[i] * b[i];
      a_mag += a[i] * a[i];
      b_mag += b[i] * b[i];
    }

    return 1 - dot_product / (std::sqrt(a_mag) * std::sqrt(b_mag));
  }

  static inline float l2Distance(const float* a, const float* b, size_t dim) {
    float dist = 0.0;

    for (size_t i = 0; i < dim; i++) {
      float delta = a[i] - b[i];
      dist += delta * delta;
    }

    return std::sqrt(dist);
  }

  static inline float innerProductDistance(const float* a, const float* b,
                                           size_t dim) {
    float dot_product = 0.0;

    for (size_t i = 0; i < dim; i++) {
      dot_product += a[i] * b[i];
    }

    return -dot_product;
  }

  uint32_t* nbrStart(size_t node) {
    assert(node < _max_nodes);
    return _edges.data() + node * _max_nbrs;
  }

  const uint32_t* nbrStart(size_t node) const {
    assert(node < _max_nodes);
    return _edges.data() + node * _max_nbrs;
  }

  uint32_t* nbrEnd(size_t node) {
    assert(node < _max_nodes);
    return _edges.data() + (node + 1) * _max_nbrs;
  }

  const uint32_t* nbrEnd(size_t node) const {
    assert(node < _max_nodes);
    return _edges.data() + (node + 1) * _max_nbrs;
  }

  const float* data(size_t node) const {
    assert(node < _max_nodes);
    return _data + node * _dim;
  }

  std::vector<uint32_t> _edges;
  size_t _curr_num_nodes;
  size_t _max_nbrs;
  size_t _dim;
  size_t _max_nodes;
  const float* _data;

  std::atomic_size_t _visited_count;
  std::atomic_size_t _n_queries;
};

}  // namespace thirdai::search::hnsw