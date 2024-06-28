#pragma once

#include <gtest/gtest.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <random>

namespace thirdai::search::tests {

template <class Index>
inline void checkQuery(const Index& index, const std::string& query,
                       const std::vector<DocId>& expected_ids) {
  auto results = index.query(query, expected_ids.size(), true);
  ASSERT_EQ(results.size(), expected_ids.size());
  for (size_t i = 0; i < expected_ids.size(); i++) {
    ASSERT_EQ(results.at(i).first, expected_ids.at(i));
  }
}

template <class Index>
inline void checkRank(const Index& index, const std::string& query,
                      const std::unordered_set<DocId>& candidates,
                      const std::vector<DocId>& expected_ids) {
  auto results = index.rank(query, candidates, expected_ids.size(), true);
  ASSERT_EQ(results.size(), expected_ids.size());
  for (size_t i = 0; i < expected_ids.size(); i++) {
    ASSERT_EQ(results.at(i).first, expected_ids.at(i));
  }
}

inline std::tuple<std::vector<DocId>, std::vector<std::string>,
                  std::vector<std::string>>
makeDocsAndQueries(size_t vocab_size, size_t n_docs) {
  std::uniform_int_distribution<> doc_length_dist(20, 70);
  std::uniform_int_distribution<> query_length_dist(5, 15);

  Tokens vocab(vocab_size);
  for (size_t i = 0; i < vocab_size; i++) {
    vocab[i] = std::to_string(i);
  }

  std::mt19937 rng(8248);

  std::vector<DocId> ids;
  std::vector<std::string> docs;
  std::vector<std::string> queries;
  for (size_t i = 0; i < n_docs; i++) {
    Tokens doc_tokens;
    std::sample(vocab.begin(), vocab.end(), std::back_inserter(doc_tokens),
                doc_length_dist(rng), rng);

    ids.push_back(i);
    docs.push_back(text::join(doc_tokens, " "));

    Tokens query;
    std::sample(doc_tokens.begin(), doc_tokens.end(), std::back_inserter(query),
                query_length_dist(rng), rng);
    queries.push_back(text::join(query, " "));
  }

  return {ids, docs, queries};
}

}  // namespace thirdai::search::tests