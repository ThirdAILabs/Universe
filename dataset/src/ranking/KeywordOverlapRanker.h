#pragma once

#include "QueryDocumentRanker.h"
#include <unordered_set>

namespace thirdai::dataset::ranking {

/**
 * This ranker attempts to rerank documents based on keyword overlap with the
 * query. We assume that a word is a keyword if it is longer than
 * min_word_length characters. We then combine the char-k grams of each keyword
 * and determine the overlap between the query and the document's keyword char-k
 * grams.
 *
 * Advantages of this ranker:
 *  - fast
 *  - captures word stems (swim vs swimming for example)
 *  - avoids noise reasonably well
 * Some limitations:
 *  - Assumes that only words greater than min_word_length are relevant
 *  - Biases words that are longer (since there are more char-4 grams for these)
 *
 * This is what worked best for ICML papers and Rice Docs but some of these
 * heuristics might not be optimal. We may want to try things like only
 * filtering out true stopwords, using TFIDF to inform keyword importance,
 * counting char-4 overlaps between long keywords as only having a score of 1,
 * etc.
 */
class KeywordOverlapRanker : public QueryDocumentRanker {
 public:
  KeywordOverlapRanker(bool lowercase, bool replace_punct,
                       uint32_t k_gram_length, size_t min_word_length)
      : _lowercase(lowercase),
        _replace_punct(replace_punct),
        _k_gram_length(k_gram_length),
        _min_word_length(min_word_length) {}

  std::pair<RankedIndices, Scores> rank(
      const std::string& query,
      const std::vector<std::string>& documents) final;

 private:
  std::unordered_set<std::string> findKeywords(std::string string) const;

  static float overlapScore(
      const std::unordered_set<std::string>& query_keywords,
      const std::unordered_set<std::string>& doc_keywords);

  bool _lowercase;
  bool _replace_punct;
  uint32_t _k_gram_length;
  size_t _min_word_length;
};

}  // namespace thirdai::dataset::ranking