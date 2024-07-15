#include "InvertedIndexTestUtils.h"
#include <gtest/gtest.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/OnDiskIndex.h>
#include <cstdlib>
#include <filesystem>

namespace thirdai::search::tests {

class OnDiskIndexTests : public ::testing::Test {
 public:
  void TearDown() final {
    for (const auto& db : _dbs_created) {
      std::filesystem::remove_all(db);
    }
  }

  std::string tmpDbName() {
    std::string name = "tmp_db_" + std::to_string(_dbs_created.size()) + "_" +
                       std::to_string(rand());
    _dbs_created.push_back(name);
    return name;
  }

 private:
  std::vector<std::string> _dbs_created;
};

TEST_F(OnDiskIndexTests, BasicRetrieval) {
  std::string db_name = tmpDbName();
  OnDiskIndex index(db_name);

  index.index({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {{"a b c d e g"},
                                                        {"a b c d"},
                                                        {"1 2 3"},
                                                        {"x y z"},
                                                        {"2 3"},
                                                        {"c f"},
                                                        {"f g d g"},
                                                        {"c d e f"},
                                                        {"f t q v w"},
                                                        {"f m n o p"},
                                                        {"f g h i"},
                                                        {"c 7 8 9 10 11"}});

  // Docs 2 and 1 both contain the whole query, but doc 2 is shorter so it ranks
  // higher. Docs 6 and 8 both contain "c" but 6 is shorter so the query terms
  // are more frequent within it.
  checkQuery(index, {"a b c"}, {2, 1, 6, 8});
  // These candidates are a subset of the original results, plus 12 which
  // usually would score lower and not be returned, but is returned when we
  // restrict the candidates. Doc 3 is also added but scores 0.
  checkRank(index, {"a b c"}, {8, 12, 3, 1}, {1, 8, 12});

  // Docs 7 and 11 contain the whole query, but 7 contains "g" repeated so it
  // scores higher. Docs 6, 8, 1 contain 1 term of the query. However 1 contains
  // "g" which occurs in fewer docs so it ranks higher. Between 6 and 8, 6 is
  // shorter so the query terms are more frequent within it.
  checkQuery(index, {"f g"}, {7, 11, 1, 6, 8});
  // These candidates are a subset of the original results plus docs 5 & 2 which
  // score 0 are added to test they are not returned.
  checkRank(index, {"f g"}, {8, 5, 6, 2, 7}, {7, 6, 8});
}

TEST_F(OnDiskIndexTests, LessFrequentTokensScoreHigher) {
  std::string db_name = tmpDbName();
  OnDiskIndex index(db_name);

  index.index({1, 2, 3, 4, 5, 6, 7}, {
                                         {"a b c d"},  // 2 query tokens
                                         {"a c f d"},  // 1 query token
                                         {"b f g k"},  // 1 query token
                                         {"a d f h"},  // 2 query tokens
                                         {"b e g e"},  // 1 query token
                                         {"h j f e"},  // 2 query tokens
                                         {"w k z m"},  // 0 query token
                                     });

  // "a" and "b" occur 4 times, "h" occurs twice, and "j" occurs once.
  // No doc contains more than 2 tokens of the query. Since doc 6 contains "h"
  // and "j" it is better than doc 4 which contains "a" and "h", which is better
  // than doc 1 which contains "a" and "b". This ordering is based on
  // prioritizing less frequent tokens.
  checkQuery(index, {"a b h j"}, {6, 4, 1});
}

TEST_F(OnDiskIndexTests, RepeatedTokensInDocs) {
  std::string db_name = tmpDbName();
  OnDiskIndex index(db_name);

  index.index(
      {1, 2, 3, 4, 5},
      {{"c a z a"}, {"y r q z"}, {"e c c m"}, {"l b f h"}, {"a b q d"}});

  // All of the tokens in the query occur in 2 docs. Doc 1 contains 2 tokens
  // from the query but one occurs twice. Doc 5 contains 2 unique tokens from
  // the query. Doc 3 contains 1 token from the query but it occurs twice. This
  // checks that if two docs have the same number unique tokens in the query,
  // but one has multiple occurences of a token from the query, then it is
  // ranked higher. This also checks that having more unique tokens is
  // preferable to have the same token repeated.
  checkQuery(index, {"c a q"}, {1, 5, 3});
}

TEST_F(OnDiskIndexTests, RepeatedTokensInQuery) {
  std::string db_name = tmpDbName();
  OnDiskIndex index(db_name);

  index.index(
      {1, 2, 3, 4, 5},
      {{"y r q z"}, {"c a z m"}, {"e c c m"}, {"a b q d"}, {"l b f h q"}});

  // All of the tokens in the query occur in 2 docs. Doc 4 has tokens "a" and
  // "q" from the query, doc 2 has tokens "a m" from the query. Doc 4 scores
  // higher because token "q" occurs more in the query than token "m".
  checkQuery(index, {"q a q m"}, {4, 2});
}

TEST_F(OnDiskIndexTests, ShorterDocsScoreHigherWithSameTokens) {
  std::string db_name = tmpDbName();
  OnDiskIndex index(db_name);

  index.index({1, 2, 3, 4, 5},
              {{"x w z k"}, {"e c a"}, {"a b c d"}, {"l b f h"}, {"y r s"}});

  // Both docs 2 and 3 contain 2 query tokens, but they form a higher fraction
  // within 2 than 3.
  checkQuery(index, {"c a q"}, {2, 3});
}

TEST_F(OnDiskIndexTests, DocRemoval) {
  std::string db_name = tmpDbName();
  OnDiskIndex index(db_name);

  index.index({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {{"a b c d e"},
                                                {"a b c d"},
                                                {"a b c"},
                                                {"a b"},
                                                {"a"},
                                                {},
                                                {},
                                                {},
                                                {},
                                                {}});

  checkQuery(index, {"a b c d e"}, {1, 2, 3, 4, 5});
  index.remove({2, 4});
  checkQuery(index, {"a b c d e"}, {1, 3, 5});
}

TEST_F(OnDiskIndexTests, SyntheticDataset) {
  size_t vocab_size = 10000;
  size_t n_docs = 1000;
  size_t topk = 10;

  auto [ids, docs, queries] = makeDocsAndQueries(vocab_size, n_docs);

  OnDiskIndex index(tmpDbName());

  index.index(ids, docs);

  std::vector<std::vector<DocScore>> original_results;
  for (size_t i = 0; i < queries.size(); i++) {
    auto results = index.query(queries[i], topk, true);
    // i-th query goes to i-th doc.
    ASSERT_EQ(results[0].first, i);

    original_results.push_back(results);
  }

  // Check that building index incrementally gets the same results.
  OnDiskIndex incremental_index(tmpDbName());

  size_t n_chunks = 10;
  size_t chunksize = n_docs / n_chunks;
  for (int i = 0; i < n_chunks; i++) {
    size_t start = i * chunksize;
    size_t end = start + chunksize;
    incremental_index.index({ids.begin() + start, ids.begin() + end},
                            {docs.begin() + start, docs.begin() + end});
  }

  for (size_t i = 0; i < queries.size(); i++) {
    ASSERT_EQ(original_results[i],
              incremental_index.query(queries[i], topk, true));
  }
}

TEST_F(OnDiskIndexTests, SaveLoad) {
  size_t vocab_size = 1000;
  size_t n_docs = 100;

  auto [ids, docs, queries] = makeDocsAndQueries(vocab_size, n_docs);

  OnDiskIndex index(tmpDbName());

  index.index({ids.begin(), ids.begin() + n_docs / 2},
              {docs.begin(), docs.begin() + n_docs / 2});

  auto original_partial_results = index.queryBatch(queries, /*k=*/5);

  std::string save_path = tmpDbName();
  index.save(save_path);

  index.index({ids.begin() + n_docs / 2, ids.end()},
              {docs.begin() + n_docs / 2, docs.end()});
  auto original_full_results = index.queryBatch(queries, /*k=*/5);

  auto loaded_index = OnDiskIndex::load(save_path, false);

  auto loaded_partial_results = loaded_index->queryBatch(queries, /*k=*/5);

  ASSERT_EQ(original_partial_results, loaded_partial_results);

  loaded_index->index({ids.begin() + n_docs / 2, ids.end()},
                      {docs.begin() + n_docs / 2, docs.end()});
  auto loaded_full_results = loaded_index->queryBatch(queries, /*k=*/5);

  ASSERT_EQ(original_full_results, loaded_full_results);
}

}  // namespace thirdai::search::tests