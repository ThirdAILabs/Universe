#include "InvertedIndexTestUtils.h"
#include <gtest/gtest.h>
#include <search/src/neural_db/on_disk/OnDiskNeuralDB.h>
#include <filesystem>
#include <optional>

namespace thirdai::search::ndb::tests {

class OnDiskNeuralDbTests : public ::testing::Test {
 public:
  OnDiskNeuralDbTests() { _prefix = search::tests::randomPath() + "_"; }

  void TearDown() final {
    for (const auto& db : _dbs_created) {
      std::filesystem::remove_all(db);
    }
  }

  std::string tmpDbName() {
    std::string name = _prefix + std::to_string(_dbs_created.size()) + ".db";
    _dbs_created.push_back(name);
    return name;
  }

 private:
  std::string _prefix;
  std::vector<std::string> _dbs_created;
};

void checkNdbQuery(OnDiskNeuralDB& db, const std::string& query,
                   const std::vector<ChunkId>& expected_ids) {
  auto results = db.query(query, expected_ids.size());
  ASSERT_EQ(results.size(), expected_ids.size());
  for (size_t i = 0; i < expected_ids.size(); i++) {
    ASSERT_EQ(results.at(i).first.id, expected_ids.at(i));
  }
}

void checkNdbRank(OnDiskNeuralDB& db, const std::string& query,
                  const QueryConstraints& constraints,
                  const std::vector<ChunkId>& expected_ids) {
  auto results = db.rank(query, constraints, expected_ids.size());
  ASSERT_EQ(results.size(), expected_ids.size());
  for (size_t i = 0; i < expected_ids.size(); i++) {
    ASSERT_EQ(results.at(i).first.id, expected_ids.at(i));
  }
}

TEST_F(OnDiskNeuralDbTests, BasicRetrieval) {
  OnDiskNeuralDB db(tmpDbName());

  db.insert("doc1", {"a b c d e g", "a b c d", "1 2 3"},
            {{{"q1", MetadataValue::Bool(true)}},
             {{"q2", MetadataValue::Bool(true)}},
             {{"q1", MetadataValue::Bool(true)}}},
            std::nullopt);

  db.insert(
      "doc2", {"x y z", "2 3", "c f", "f g d g", "c d e f"},
      {{},
       {{"q2", MetadataValue::Bool(true)}},
       {{"q2", MetadataValue::Bool(true)}},
       {{"q2", MetadataValue::Bool(true)}},
       {{"q1", MetadataValue::Bool(true)}, {"q2", MetadataValue::Bool(true)}}},
      std::nullopt);

  db.insert("doc3", {"f t q v w", "f m n o p", "f g h i", "c 7 8 9 10 11"},
            {{}, {}, {}, {{"q1", MetadataValue::Bool(true)}}}, std::nullopt);

  // Docs 2 and 1 both contain the whole query, but doc 2 is shorter so it
  // ranks higher. Docs 6 and 8 both contain "c" but 6 is shorter so the
  // query terms are more frequent within it.
  checkNdbQuery(db, {"a & b c"}, {1, 0, 5, 7});
  // These candidates are a subset of the original results, plus 12 which
  // usually would score lower and not be returned, but is returned when we
  // restrict the candidates. Doc 3 is also added but scores 0.
  checkNdbRank(db, {"a b c"},
               {{"q1", EqualTo::make(MetadataValue::Bool(true))}}, {0, 7, 11});

  // Docs 7 and 11 contain the whole query, but 7 contains "g" repeated so it
  // scores higher. Docs 6, 8, 1 contain 1 term of the query. However 1 contains
  // "g" which occurs in fewer docs so it ranks higher. Between 6 and 8, 6 is
  // shorter so the query terms are more frequent within it.
  checkNdbQuery(db, {"f g"}, {6, 10, 0, 5, 7});
  // These candidates are a subset of the original results plus docs 5 & 2 which
  // score 0 are added to test they are not returned.
  checkNdbRank(db, {"f g"}, {{"q2", EqualTo::make(MetadataValue::Bool(true))}},
               {6, 5, 7});
}

TEST_F(OnDiskNeuralDbTests, LessFrequentTokensScoreHigher) {
  OnDiskNeuralDB db(tmpDbName());

  db.insert("doc",
            {
                "a b c d",  // 2 query tokens
                "a c f d",  // 1 query token
                "b f g k",  // 1 query token
                "a d f h",  // 2 query tokens
                "b e g e",  // 1 query token
                "h j f e",  // 2 query tokens
                "w k z m",  // 0 query token
            },
            {{}, {}, {}, {}, {}, {}, {}}, std::nullopt);

  // "a" and "b" occur 4 times, "h" occurs twice, and "j" occurs once.
  // No doc contains more than 2 tokens of the query. Since doc 6 contains "h"
  // and "j" it is better than doc 4 which contains "a" and "h", which is better
  // than doc 1 which contains "a" and "b". This ordering is based on
  // prioritizing less frequent tokens.
  checkNdbQuery(db, {"a b h j"}, {5, 3, 0});
}

TEST_F(OnDiskNeuralDbTests, RepeatedTokensInDocs) {
  OnDiskNeuralDB db(tmpDbName());

  db.insert("doc", {"c a z a", "y r q z", "e c c m", "l b f h", "a b q d"},
            {{}, {}, {}, {}, {}}, std::nullopt);

  // All of the tokens in the query occur in 2 docs. Doc 4 has tokens "a" and
  // "q" from the query, doc 2 has tokens "a m" from the query. Doc 4 scores
  // higher because token "q" occurs more in the query than token "m".
  checkNdbQuery(db, {"c a q"}, {0, 4, 2});
}

TEST_F(OnDiskNeuralDbTests, RepeatedTokensInQuery) {
  OnDiskNeuralDB db(tmpDbName());

  db.insert("doc", {"y r q z", "c a z m", "e c c m", "a b q d", "l b f h q"},
            {{}, {}, {}, {}, {}}, std::nullopt);

  // All of the tokens in the query occur in 2 docs. Doc 4 has tokens "a" and
  // "q" from the query, doc 2 has tokens "a m" from the query. Doc 4 scores
  // higher because token "q" occurs more in the query than token "m".
  checkNdbQuery(db, {"q a q m"}, {3, 1});
}

TEST_F(OnDiskNeuralDbTests, ShorterDocsScoreHigherWithSameTokens) {
  OnDiskNeuralDB db(tmpDbName());

  db.insert("doc", {"x w z k", "e c a", "a b c d", "l b f h", "y r s"},
            {{}, {}, {}, {}, {}}, std::nullopt);
  // Both docs 2 and 3 contain 2 query tokens, but they form a higher fraction
  // within 2 than 3.
  checkNdbQuery(db, {"c a q"}, {1, 2});
}

TEST_F(OnDiskNeuralDbTests, ConstrainedSearch) {
  OnDiskNeuralDB db(tmpDbName());

  db.insert("doc",
            {"a", "a b", "a b c", "a b c d", "a b c d e", "w", "w x", "w x y",
             "w x y z"},
            {{
                 {"k1", MetadataValue::Float(5.2)},
                 {"k2", MetadataValue::Bool(false)},
                 {"k3", MetadataValue::Int(7)},
                 {"k4", MetadataValue::Str("apple")},
             },
             {
                 // Eliminated by constraint 2
                 {"k1", MetadataValue::Float(3.1)},
                 {"k2", MetadataValue::Bool(true)},
                 {"k3", MetadataValue::Int(22)},
                 {"k4", MetadataValue::Str("banana")},
             },
             {
                 // Eliminated by constraint 3
                 {"k1", MetadataValue::Float(9.2)},
                 {"k2", MetadataValue::Bool(false)},
                 {"k3", MetadataValue::Int(11)},
                 {"k4", MetadataValue::Str("kiwi")},
             },
             {
                 // Eliminated by constraint 1
                 {"k1", MetadataValue::Float(2.9)},
                 {"k2", MetadataValue::Bool(false)},
                 {"k3", MetadataValue::Int(7)},
                 {"k4", MetadataValue::Str("grape")},
             },
             {
                 // Eliminated by constraint 4
                 {"k1", MetadataValue::Float(4.7)},
                 {"k2", MetadataValue::Bool(false)},
                 {"k3", MetadataValue::Int(22)},
                 {"k4", MetadataValue::Str("pineapple")},
             },
             {},
             {},
             {},
             {}},
            std::nullopt);

  checkNdbQuery(db, "a b c d e", {4, 3, 2, 1, 0});

  QueryConstraints constraints = {
      {"k1", GreaterThan::make(MetadataValue::Float(3))},
      {"k2", EqualTo::make(MetadataValue::Bool(false))},
      {"k3", AnyOf::make({MetadataValue::Int(7), MetadataValue::Int(22)})},
      {"k4", LessThan::make(MetadataValue::Str("peach"))},
  };

  checkNdbRank(db, "a b c d e", constraints, {0});
}

std::vector<std::vector<std::pair<Chunk, float>>> queryDb(
    OnDiskNeuralDB& db, const std::vector<std::string>& queries,
    uint32_t topk) {
  std::vector<std::vector<std::pair<Chunk, float>>> results;
  for (const auto& query : queries) {
    auto chunks = db.query(query, topk);
    results.push_back(chunks);
  }
  return results;
}

static void compareChunkLists(const std::vector<std::pair<Chunk, float>>& a,
                              const std::vector<std::pair<Chunk, float>>& b) {
  ASSERT_EQ(a.size(), b.size());

  for (size_t i = 0; i < a.size(); i++) {
    ASSERT_EQ(a[i].first.id, b[i].first.id);
    ASSERT_EQ(a[i].first.text, b[i].first.text);
    ASSERT_FLOAT_EQ(a[i].second, b[i].second);
  }
}

static void compareResults(std::vector<std::pair<Chunk, float>> a,
                           std::vector<std::pair<Chunk, float>> b) {
  // For some queries two docs may have the same score. For different numbers of
  // shards the docs may have a different ordering when the score is the
  // same. Sorting by doc ids if the scores are the same solves this, it only
  // doesn't handle if a doc doesn't make the topk cuttoff because of this.
  // Removing the last item by allowing the end to differ as long as the prior
  // results match.

  auto sort = [](auto& vec) {
    std::sort(vec.begin(), vec.end(), [](const auto& x, const auto& y) {
      if (x.second == y.second) {
        return x.first.id < y.first.id;
      }
      return x.second > y.second;
    });
  };

  sort(a);
  sort(b);

  a.pop_back();
  b.pop_back();

  compareChunkLists(a, b);
}

TEST_F(OnDiskNeuralDbTests, SyntheticDataset) {
  size_t vocab_size = 10000;
  size_t n_docs = 1000;
  size_t topk = 10;

  auto [_, docs, queries] =
      search::tests::makeDocsAndQueries(vocab_size, n_docs);

  OnDiskNeuralDB db(tmpDbName());

  db.insert("doc", docs, std::vector<MetadataMap>(docs.size(), MetadataMap()),
            std::nullopt);

  const auto results = queryDb(db, queries, topk);

  for (size_t i = 0; i < queries.size(); i++) {
    // i-th query goes to i-th doc.
    ASSERT_EQ(results[i][0].first.id, i);
    // Check single query vs batch query consistency.
    compareChunkLists(db.query(queries[i], /*top_k=*/topk), results[i]);
  }

  OnDiskNeuralDB incremental_db(tmpDbName());

  // Check that building index incrementally gets the same results.
  size_t n_chunks = 10;
  size_t chunksize = n_docs / n_chunks;
  for (int i = 0; i < n_chunks; i++) {
    size_t start = i * chunksize;
    size_t end = start + chunksize;
    incremental_db.insert(
        "doc" + std::to_string(i), {docs.begin() + start, docs.begin() + end},
        std::vector<MetadataMap>(end - start, MetadataMap()), std::nullopt);
  }

  auto incremental_results = queryDb(incremental_db, queries, topk);

  ASSERT_EQ(results.size(), incremental_results.size());
  for (size_t i = 0; i < results.size(); i++) {
    compareResults(results[i], incremental_results[i]);
  }
}

}  // namespace thirdai::search::ndb::tests