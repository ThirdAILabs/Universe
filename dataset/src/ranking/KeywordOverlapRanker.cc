#include "KeywordOverlapRanker.h"
#include <utils/StringManipulation.h>
#include <algorithm>
#include <string>

namespace thirdai::dataset::ranking {

std::pair<RankedIndices, Scores> KeywordOverlapRanker::rank(
    const std::string& query, const std::vector<std::string>& documents) {
  std::unordered_set<std::string> query_keywords = findKeywords(query);

  std::vector<std::pair<float, uint32_t>> score_pairs;
  for (size_t i = 0; i < documents.size(); i++) {
    std::unordered_set<std::string> doc_keywords = findKeywords(documents[i]);
    score_pairs.push_back({overlapScore(query_keywords, doc_keywords), i});
  }

  std::sort(score_pairs.begin(), score_pairs.end(),
            [](auto pair1, auto pair2) { return pair1.first > pair2.first; });

  Scores sorted_scores;
  RankedIndices ranked_indices;
  for (const auto& [score, index] : score_pairs) {
    sorted_scores.push_back(score);
    ranked_indices.push_back(index);
  }

  return {std::move(ranked_indices), std::move(sorted_scores)};
}

std::unordered_set<std::string> KeywordOverlapRanker::findKeywords(
    std::string string) const {
  if (_lowercase) {
    string = text::lower(string);
  }

  if (_replace_punct) {
    string = text::replacePunctuation(string);
  }

  std::vector<std::string> words = text::split(string, /* delimiter= */ ' ');

  std::vector<std::string> word_char_k_grams =
      text::wordLevelCharKGrams(words, /* k=*/_k_gram_length,
                                /* min_word_length=*/_min_word_length);

  std::unordered_set<std::string> unique_char_k_grams(word_char_k_grams.begin(),
                                                      word_char_k_grams.end());

  return unique_char_k_grams;
}

float KeywordOverlapRanker::overlapScore(
    const std::unordered_set<std::string>& query_keywords,
    const std::unordered_set<std::string>& doc_keywords) {
  uint32_t value = 0;
  for (const auto& token : query_keywords) {
    if (doc_keywords.count(token)) {
      value += 1;
    }
  }
  return value;
}

}  // namespace thirdai::dataset::ranking