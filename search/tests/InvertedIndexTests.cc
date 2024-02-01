#include <gtest/gtest.h>
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
  InvertedIndex index(1.0);

  index.index({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
              {{"a", "b", "c", "d", "e", "g"},
               {"a", "b", "c", "d"},
               {"1", "2", "3"},
               {"x", "y", "z"},
               {"2", "3"},
               {"c", "f"},
               {"f", "g", "d", "g"},
               {"c", "d", "e", "f"},
               {"t", "q", "v"},
               {"m", "n", "o"},
               {"f", "g", "h", "i"}});

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
  InvertedIndex index(1.0);

  index.index({1, 2, 3, 4, 5, 6, 7},
              {
                  {"a", "b", "c", "d"},  // 2 query tokens
                  {"a", "c", "f", "d"},  // 1 query token
                  {"b", "f", "g", "k"},  // 1 query token
                  {"a", "d", "f", "h"},  // 2 query tokens
                  {"b", "e", "g", "e"},  // 1 query token
                  {"h", "j", "f", "e"},  // 2 query tokens
                  {"w", "k", "z", "m"},  // 0 query token
              });

  // "a" and "b" occur 4 times, "h" occurs twice, and "j" occurs once.
  // No doc contains more than 2 tokens of the query. Since doc 6 contains "h"
  // and "j" it is better than doc 4 which contains "a" and "h", which is better
  // than doc 1 which contains "a" and "b". This ordering is based on
  // prioritizing less frequent tokens.
  checkQuery(index, {"a", "b", "h", "j"}, {6, 4, 1});
}

TEST(InvertedIndexTests, RepeatedTokensInDocs) {
  InvertedIndex index(1.0);

  index.index({1, 2, 3, 4, 5}, {{"c", "a", "z", "a"},
                                {"y", "r", "q", "z"},
                                {"e", "c", "c", "m"},
                                {"l", "b", "f", "h"},
                                {"a", "b", "q", "d"}});

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
  InvertedIndex index(1.0);

  index.index({1, 2, 3, 4, 5}, {{"y", "r", "q", "z"},
                                {"c", "a", "z", "m"},
                                {"e", "c", "c", "m"},
                                {"a", "b", "q", "d"},
                                {"l", "b", "f", "h"}});

  // All of the tokens in the query occur in 2 docs. Doc 4 has tokens "a" and
  // "q" from the query, doc 2 has tokens "a", "m" from the query. Doc 4 scores
  // higher because token "q" occurs more in the query than token "m".
  checkQuery(index, {"q", "a", "q", "m"}, {4, 2});
}

TEST(InvertedIndexTests, ShorterDocsScoreHigherWithSameTokens) {
  InvertedIndex index(1.0);

  index.index({1, 2, 3, 4, 5}, {{"x", "w", "z", "k"},
                                {"e", "c", "a"},
                                {"a", "b", "c", "d"},
                                {"l", "b", "f", "h"},
                                {"y", "r", "s"}});

  // Both docs 2 and 3 contain 2 query tokens, but they form a higher fraction
  // within 2 than 3.
  checkQuery(index, {"c", "a", "q"}, {2, 3});
}

std::tuple<std::vector<DocId>, std::vector<Tokens>, std::vector<Tokens>>
makeDocsAndQueries(size_t vocab_size, size_t n_docs) {
  std::uniform_int_distribution<> doc_length_dist(20, 70);
  std::uniform_int_distribution<> query_length_dist(5, 15);

  Tokens vocab(vocab_size);
  for (size_t i = 0; i < vocab_size; i++) {
    vocab[i] = std::to_string(i);
  }

  std::mt19937 rng(8248);

  std::vector<DocId> ids;
  std::vector<Tokens> docs;
  std::vector<Tokens> queries;
  for (size_t i = 0; i < n_docs; i++) {
    Tokens doc_tokens;
    std::sample(vocab.begin(), vocab.end(), std::back_inserter(doc_tokens),
                doc_length_dist(rng), rng);

    ids.push_back(i);
    docs.push_back(doc_tokens);

    Tokens query;
    std::sample(doc_tokens.begin(), doc_tokens.end(), std::back_inserter(query),
                query_length_dist(rng), rng);
    queries.push_back(query);
  }

  return {ids, docs, queries};
}

TEST(InvertedIndexTests, SyntheticDataset) {
  size_t vocab_size = 10000;
  size_t n_docs = 1000;

  auto [ids, docs, queries] = makeDocsAndQueries(vocab_size, n_docs);

  InvertedIndex index;
  index.index(ids, docs);

  auto results = index.queryBatch(queries, /*k=*/5);

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
    size_t start = i * chunksize;
    size_t end = start + chunksize;
    incremental_index.index({ids.begin() + start, ids.begin() + end},
                            {docs.begin() + start, docs.begin() + end});
  }

  auto incremental_results = incremental_index.queryBatch(queries, /*k=*/5);

  ASSERT_EQ(results, incremental_results);
}

TEST(InvertedIndexTests, SaveLoad) {
  size_t vocab_size = 1000;
  size_t n_docs = 100;

  auto [ids, docs, queries] = makeDocsAndQueries(vocab_size, n_docs);

  InvertedIndex index;
  index.index({ids.begin(), ids.begin() + n_docs / 2},
              {docs.begin(), docs.begin() + n_docs / 2});
  auto original_partial_results = index.queryBatch(queries, /*k=*/5);

  std::string save_path = "./test_partial_index";
  index.save(save_path);

  index.index({ids.begin() + n_docs / 2, ids.end()},
              {docs.begin() + n_docs / 2, docs.end()});
  auto original_full_results = index.queryBatch(queries, /*k=*/5);

  auto loaded_index = InvertedIndex::load(save_path);

  auto loaded_partial_results = loaded_index->queryBatch(queries, /*k=*/5);

  ASSERT_EQ(original_partial_results, loaded_partial_results);

  loaded_index->index({ids.begin() + n_docs / 2, ids.end()},
                      {docs.begin() + n_docs / 2, docs.end()});
  auto loaded_full_results = loaded_index->queryBatch(queries, /*k=*/5);

  ASSERT_EQ(original_full_results, loaded_full_results);
}

}  // namespace thirdai::search::tests