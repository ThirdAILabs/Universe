#include <gtest/gtest.h>
#include <_types/_uint64_t.h>
#include <search/src/InvertedIndex.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <vector>

namespace thirdai::search::tests {

void checkQuery(const InvertedIndex& index, const Tokens& query,
                const std::vector<DocId>& expected_ids) {
  auto results = index.query(query, expected_ids.size());
  ASSERT_EQ(results.size(), expected_ids.size());
  for (size_t i = 0; i < expected_ids.size(); i++) {
    ASSERT_EQ(results.at(i).first, expected_ids.at(i));
  }
}

TEST(InvertedIndexTests, BasicRetrieval) {
  InvertedIndex index;

  index.index({
      {1, {"a", "b", "c", "d", "e", "g"}},
      {2, {"a", "b", "c", "d"}},
      {3, {"1", "2", "3"}},
      {4, {"x", "y", "z"}},
      {5, {"2", "3"}},
      {6, {"c", "f"}},
      {7, {"f", "g", "d", "g"}},
      {8, {"c", "d", "e", "f"}},
      {9, {"t", "q", "v"}},
      {10, {"m", "n", "o"}},
      {11, {"f", "g", "h", "i"}},
  });

  // Docs 2 and 1 both contain the whole query, but doc 2 is shorter so it ranks
  // higher. Docs 6 and 8 both contain "c" but 6 is shorter so the query terms
  // are more frequent within it.
  checkQuery(index, {"a", "b", "c"}, {2, 1, 6, 8});

  // Docs 7 and 11 contain the whole query, but 7 contains "g" repeated so it
  // scores higher. Docs 6, 8, 1 contain 1 term of the query. However 1 contains
  // "g" which occurs in fewer docs so it ranks higher. Between 6 and 8, 6 is
  // shorter so the query terms are more frequent within it.
  checkQuery(index, {"f", "g"}, {7, 11, 1, 6, 8});
}

TEST(InvertedIndexTests, LessFrequentTokensScoreHigher) {
  InvertedIndex index;

  index.index({
      {1, {"a", "b", "c", "d"}},  // 2 query tokens
      {2, {"a", "c", "f", "d"}},  // 1 query token
      {3, {"b", "f", "g", "k"}},  // 1 query token
      {4, {"a", "d", "f", "h"}},  // 2 query tokens
      {5, {"b", "e", "g", "e"}},  // 1 query token
      {6, {"h", "j", "f", "e"}},  // 2 query tokens
      {7, {"w", "k", "z", "m"}},  // 0 query token
  });

  // "a" and "b" occur 4 times, "h" occurs twice, and "j" occurs once.
  // No doc contains more than 2 tokens of the query. Since doc 6 contains "h"
  // and "j" it is better than doc 4 which contains "a" and "h", which is better
  // than doc 1 which contains "a" and "b". This ordering is based on
  // prioritizing less frequent tokens.
  checkQuery(index, {"a", "b", "h", "j"}, {6, 4, 1});
}

TEST(InvertedIndexTests, RepeatedTokensInDocs) {
  InvertedIndex index;

  index.index({
      {1, {"c", "a", "z", "a"}},
      {2, {"y", "r", "q", "z"}},
      {3, {"e", "c", "c", "m"}},
      {4, {"l", "b", "f", "h"}},
      {5, {"a", "b", "q", "d"}},
  });

  // All of the tokens in the query occur in 2 docs. Doc 1 contains 2 tokens
  // from the query but one occurs twice. Doc 5 contains 2 unique tokens from
  // the query. Doc 3 contains 1 token from the query but it occurs twice. This
  // checks that if two docs have the same number unique tokens in the query,
  // but one has multiple occurences of a token from the query, then it is
  // ranked higher. This also checks that having more unique tokens is
  // preferable to have the same token repeated.
  checkQuery(index, {"c", "a", "q"}, {1, 5, 3});
}

TEST(InvertedIndexTests, RepeatedTokensInQuery) {
  InvertedIndex index;

  index.index({
      {1, {"y", "r", "q", "z"}},
      {2, {"c", "a", "z", "m"}},
      {3, {"e", "c", "c", "m"}},
      {4, {"a", "b", "q", "d"}},
      {5, {"l", "b", "f", "h"}},
  });

  // All of the tokens in the query occur in 2 docs. Doc 4 has tokens "a" and
  // "q" from the query, doc 2 has tokens "a", "m" from the query. Doc 4 scores
  // higher because token "q" occurs more in the query than token "m".
  checkQuery(index, {"q", "a", "q", "m"}, {4, 2});
}

TEST(InvertedIndexTests, ShorterDocsScoreHigherWithSameTokens) {
  InvertedIndex index;

  index.index({
      {1, {"x", "w", "z", "k"}},
      {2, {"e", "c", "a"}},
      {3, {"a", "b", "c", "d"}},
      {4, {"l", "b", "f", "h"}},
      {5, {"y", "r", "s"}},
  });

  // Both docs 2 and 3 contain 2 query tokens, but they form a higher fraction
  // within 2 than 3.
  checkQuery(index, {"c", "a", "q"}, {2, 3});
}

std::pair<std::vector<std::pair<DocId, Tokens>>, std::vector<Tokens>>
makeDocsAndQueries(size_t vocab_size, size_t n_docs) {
  std::uniform_int_distribution<> doc_length_dist(20, 70);
  std::uniform_int_distribution<> query_length_dist(5, 15);

  Tokens vocab(vocab_size);
  for (size_t i = 0; i < vocab_size; i++) {
    vocab[i] = std::to_string(i);
  }

  std::mt19937 rng(8248);

  std::vector<std::pair<DocId, Tokens>> docs;
  std::vector<Tokens> queries;
  for (size_t i = 0; i < n_docs; i++) {
    Tokens doc_tokens;
    std::sample(vocab.begin(), vocab.end(), std::back_inserter(doc_tokens),
                doc_length_dist(rng), rng);

    docs.emplace_back(i, doc_tokens);

    Tokens query;
    std::sample(doc_tokens.begin(), doc_tokens.end(), std::back_inserter(query),
                query_length_dist(rng), rng);
    queries.push_back(query);
  }

  return {docs, queries};
}

TEST(InvertedIndexTests, SyntheticDataset) {
  size_t vocab_size = 10000;
  size_t n_docs = 1000;

  auto [docs, queries] = makeDocsAndQueries(vocab_size, n_docs);

  InvertedIndex index;
  index.index(docs);

  auto results = index.query(queries, /*k=*/5);

  for (size_t i = 0; i < queries.size(); i++) {
    // i-th query goes to i-th doc.
    ASSERT_EQ(results[i][0].first, i);
    // Check single query vs batch query consistency.
    ASSERT_EQ(index.query(queries[i], /*k=*/5), results[i]);
  }

  // Check that building index incrementally gets the same results.
  InvertedIndex incremental_index;
  size_t n_chunks = 10;
  size_t chunksize = n_docs / n_chunks;
  for (int i = 0; i < n_chunks; i++) {
    incremental_index.index(
        {docs.begin() + i * chunksize, docs.begin() + (i + 1) * chunksize});
  }

  auto incremental_results = incremental_index.query(queries, /*k=*/5);

  ASSERT_EQ(results, incremental_results);
}

}  // namespace thirdai::search::tests