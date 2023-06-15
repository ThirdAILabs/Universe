#include "HNSW.h"
#include <bolt/src/utils/ProgressBar.h>
#include <bolt/src/utils/Timer.h>
#include <unordered_set>

namespace thirdai::search::hnsw {

HNSW::HNSW(size_t max_nbrs, size_t dim, size_t n_nodes, const float* data,
           size_t construction_buf_size, size_t num_initializations)
    : _edges(n_nodes * max_nbrs),
      _curr_num_nodes(0),
      _max_nbrs(max_nbrs),
      _dim(dim),
      _max_nodes(n_nodes),
      _data(data) {
  for (uint32_t i = 0; i < n_nodes; i++) {
    std::fill_n(nbrStart(i), _max_nbrs, i);  // Initialize as self loops.
  }

  ProgressBar bar("Constructing HNSW Index", n_nodes);

  bolt::utils::Timer timer;

  for (uint32_t i = 0; i < n_nodes; i++) {
    insert(this->data(i), construction_buf_size, num_initializations);
    bar.increment();
  }

  timer.stop();

  bar.close("Constructed hnsw index in " +
            std::to_string(timer.milliseconds() / 1000) + " seconds.");
}

std::vector<uint32_t> HNSW::query(const float* query, uint32_t k,
                                  size_t search_buffer_size,
                                  size_t num_initializations) const {
  uint32_t entry = searchInitialization(query, num_initializations);

  auto candidates = beamSearch(query, entry, search_buffer_size);

  uint32_t n_outputs = std::min<uint32_t>(k, candidates.size());
  std::vector<uint32_t> results;
  results.reserve(n_outputs);
  for (uint32_t i = 0; i < n_outputs; i++) {
    results.push_back(candidates.top().node);
    candidates.pop();
  }

  return results;
}

std::unordered_set<uint32_t> HNSW::querySet(const float* query, uint32_t k,
                                            size_t search_buffer_size,
                                            size_t num_initializations) const {
  uint32_t entry = searchInitialization(query, num_initializations);

  auto candidates = beamSearch(query, entry, search_buffer_size);

  uint32_t n_outputs = std::min<uint32_t>(k, candidates.size());
  std::unordered_set<uint32_t> results;
  results.reserve(n_outputs);
  for (uint32_t i = 0; i < n_outputs; i++) {
    results.insert(candidates.top().node);
    candidates.pop();
  }

  return results;
}

void HNSW::insert(const float* data, size_t search_buffer_size,
                  size_t num_initializations) {
  uint32_t new_node_id = _curr_num_nodes;

  if (_curr_num_nodes == 0) {
    _curr_num_nodes++;
    return;
  }

  uint32_t entry = searchInitialization(data, num_initializations);

  auto candidates = beamSearch(data, entry, search_buffer_size);

  auto neighbors = selectNeighbors(candidates);

  connectNeighbors(neighbors, new_node_id);

  _curr_num_nodes++;
}

uint32_t HNSW::searchInitialization(const float* query,
                                    size_t num_initializations) const {
  size_t step_size = std::max<size_t>(_curr_num_nodes / num_initializations, 1);

  float min_dist = std::numeric_limits<float>::max();
  uint32_t entry_node = 0;

  for (uint32_t node = 0; node < _curr_num_nodes; node += step_size) {
    float dist = distance(query, data(node));
    if (dist < min_dist) {
      min_dist = dist;
      entry_node = node;
    }
  }

  return entry_node;
}

ClosestQueue HNSW::beamSearch(const float* query, uint32_t entry_node,
                              size_t buffer_size) const {
  std::unordered_set<uint32_t> visited;
  ClosestQueue candidates;
  FurthestQueue worklist;

  float initial_dist = distance(query, data(entry_node));
  visited.insert(entry_node);
  candidates.push({entry_node, initial_dist});
  worklist.push({entry_node, initial_dist});

  while (!candidates.empty()) {
    NodeDistPair best_candidate = candidates.top();
    candidates.pop();

    if (best_candidate.dist > worklist.top().dist) {
      break;
    }

    for (const uint32_t* nbr = nbrStart(best_candidate.node);
         nbr != nbrEnd(best_candidate.node); nbr++) {
      if (!visited.count(*nbr)) {
        visited.insert(*nbr);
        float dist = distance(query, data(*nbr));

        if (worklist.size() < buffer_size || dist < worklist.top().dist) {
          candidates.push({*nbr, dist});
          worklist.push({*nbr, dist});
          if (worklist.size() > buffer_size) {
            worklist.pop();
          }
        }
      }
    }
  }

  ClosestQueue output;
  while (!worklist.empty()) {
    output.push(worklist.top());
    worklist.pop();
  }
  return output;
}

ClosestQueue HNSW::selectNeighbors(ClosestQueue& candidates) const {
  if (candidates.size() < _max_nbrs) {
    return candidates;
  }

  std::vector<NodeDistPair> selected;

  while (!candidates.empty() && selected.size() < _max_nbrs) {
    NodeDistPair best_candidate = candidates.top();
    candidates.pop();

    bool keep = true;

    /**
     * Let Q be the query node used to get the set of candidates.Let S be one of
     * the nodes already selected as a neighbor of Q, let C be the candidate
     * node we are considering adding as a neighbor. Since S has already be
     * selected as a neighbor we know that d(Q,S) <= d(Q,C). Now if we also
     * find that D(S,C) < D(Q,C) then we skip the edge (Q,C) as there is a path
     * Q->S->C that only traverses shorter edges than Q->C. Note that this
     * heuristic does not actually check that the edge S->C exists, only that in
     * theory it could exist and be shorter.
     */
    for (const auto& choosen : selected) {
      float dist = distance(data(choosen.node), data(best_candidate.node));

      if (dist < best_candidate.dist) {
        keep = false;
        break;
      }
    }
    if (keep) {
      selected.push_back(best_candidate);
    }
  }

  ClosestQueue output;
  for (auto& x : selected) {
    output.push(x);
  }
  return output;
}

void HNSW::connectNeighbors(ClosestQueue& neighbors, uint32_t new_node) {
  uint32_t nbr_idx = 0;
  while (!neighbors.empty()) {
    assert(nbr_idx <= _max_nbrs);

    uint32_t nbr_id = neighbors.top().node;
    neighbors.pop();

    nbrStart(new_node)[nbr_idx] = nbr_id;
    ++nbr_idx;

    bool added = false;
    for (uint32_t* nbr = nbrStart(nbr_id); nbr != nbrEnd(nbr_id); nbr++) {
      if (*nbr == nbr_id) {
        *nbr = new_node;
        added = true;
        break;
      }
    }

    if (added) {
      continue;
    }

    ClosestQueue candidates;
    candidates.push({new_node, distance(data(nbr_id), data(new_node))});

    for (uint32_t* nbr = nbrStart(nbr_id); nbr != nbrEnd(nbr_id); nbr++) {
      float dist = distance(data(nbr_id), data(*nbr));
      candidates.push({*nbr, dist});
    }

    candidates = selectNeighbors(candidates);

    uint32_t index = 0;
    while (!candidates.empty()) {
      nbrStart(nbr_id)[index] = candidates.top().node;
      candidates.pop();
      ++index;
    }

    for (; index < _max_nbrs; index++) {
      nbrStart(nbr_id)[index] = nbr_id;
    }
  }
}

}  // namespace thirdai::search::hnsw