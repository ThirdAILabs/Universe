#pragma once

#include <unordered_set>
#include <vector>

namespace thirdai::search {

using DocId = uint64_t;
using DocScore = std::pair<DocId, float>;

class Retriever {
 public:
  virtual void index(const std::vector<DocId>& ids,
                     const std::vector<std::string>& docs) = 0;

  virtual std::vector<DocScore> query(const std::string& query, uint32_t k,
                                      bool parallelize) const = 0;

  virtual std::vector<DocScore> rank(
      const std::string& query, const std::unordered_set<DocId>& candidates,
      uint32_t k, bool parallelize) const = 0;

  std::vector<std::vector<DocScore>> queryBatch(
      const std::vector<std::string>& queries, uint32_t k) const {
    std::vector<std::vector<DocScore>> scores(queries.size());

#pragma omp parallel for default(none) \
    shared(queries, scores, k) if (queries.size() > 1)
    for (size_t i = 0; i < queries.size(); i++) {
      scores[i] = query(queries[i], k, /*parallelize=*/false);
    }

    return scores;
  }

  std::vector<std::vector<DocScore>> rankBatch(
      const std::vector<std::string>& queries,
      const std::vector<std::unordered_set<DocId>>& candidates,
      uint32_t k) const {
    if (queries.size() != candidates.size()) {
      throw std::invalid_argument(
          "Number of queries must match number of candidate sets for ranking.");
    }

    std::vector<std::vector<DocScore>> scores(queries.size());

#pragma omp parallel for default(none) \
    shared(queries, candidates, scores, k) if (queries.size() > 1)
    for (size_t i = 0; i < queries.size(); i++) {
      scores[i] = rank(queries[i], candidates[i], k, /*parallelize=*/false);
    }

    return scores;
  }

  virtual void prune() {}

  virtual void remove(const std::vector<DocId>& ids) = 0;

  virtual size_t size() const = 0;

  virtual void save(const std::string& save_path) const = 0;

  virtual std::string type() const = 0;

  virtual ~Retriever() = default;
};

}  // namespace thirdai::search