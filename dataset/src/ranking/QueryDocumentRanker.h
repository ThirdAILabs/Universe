#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace thirdai::dataset::ranking {

using RankedIndices = std::vector<uint32_t>;
using Scores = std::vector<float>;

class QueryDocumentRanker {
 public:
  virtual std::pair<RankedIndices, Scores> rank(
      const std::string& query, const std::vector<std::string>& documents) = 0;

  virtual ~QueryDocumentRanker() = default;
};

}  // namespace thirdai::dataset::ranking