#pragma once

#include "QueryDocumentRanker.h"
#include <unordered_set>

namespace thirdai::dataset::ranking {

class KeywordOverlapRanker : public QueryDocumentRanker {
 public:
  KeywordOverlapRanker() {}

  std::tuple<RankedIndices, Scores> rank(
      const std::string& query,
      const std::vector<std::string>& documents) final;

 private:
  std::unordered_set<std::string> transform(const std::string& text);

  bool _lowercase;
};

}  // namespace thirdai::dataset::ranking