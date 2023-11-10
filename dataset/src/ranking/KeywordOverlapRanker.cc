#include "KeywordOverlapRanker.h"
#include <utils/StringManipulation.h>
#include <string>

namespace thirdai::dataset::ranking {

std::tuple<RankedIndices, Scores> KeywordOverlapRanker::rank(
    const std::string& query, const std::vector<std::string>& documents) {
  std::unordered_set<std::string> tokenized_query = transform(query);

  Scores doc_scores;
  for (const auto& doc : documents) {
    std::unordered_set<std::string> tokenized_doc = transform(doc);
    doc_scores.push_back(score(tokenized_query, tokenized_doc));
  }
}

std::unordered_set<std::string> KeywordOverlapRanker::transform(
    const std::string& string) {
  string = text::lower(string);
  string = text::replacePunctuationWithSpaces(string);
  string = text::replaceNumbers()
  string = text::split(string);
}

}  // namespace thirdai::dataset::ranking