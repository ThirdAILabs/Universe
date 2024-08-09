#include "InvertedIndexTestUtils.h"
#include "RetrieverTests.h"
#include <gtest/gtest.h>
#include <search/src/inverted_index/IndexConfig.h>
#include <search/src/inverted_index/OnDiskIndex.h>
#include <search/src/inverted_index/ShardedRetriever.h>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>

namespace thirdai::search::tests {

class OnDiskIndexTests : public ::testing::Test {
 public:
  OnDiskIndexTests() { _prefix = randomPath() + "_"; }

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

TEST_F(OnDiskIndexTests, BasicRetrieval) {
  OnDiskIndex index(tmpDbName());
  testBasicRetrieval(index);
}

TEST_F(OnDiskIndexTests, LessFrequentTokensScoreHigher) {
  OnDiskIndex index(tmpDbName());
  testLessFrequentTokensScoreHigher(index);
}

TEST_F(OnDiskIndexTests, RepeatedTokensInDocs) {
  OnDiskIndex index(tmpDbName());
  testRepeatedTokensInDocs(index);
}

TEST_F(OnDiskIndexTests, RepeatedTokensInQuery) {
  OnDiskIndex index(tmpDbName());
  testRepeatedTokensInQuery(index);
}

TEST_F(OnDiskIndexTests, ShorterDocsScoreHigherWithSameTokens) {
  OnDiskIndex index(tmpDbName());
  testShorterDocsScoreHigherWithSameTokens(index);
}

TEST_F(OnDiskIndexTests, DocRemoval) {
  OnDiskIndex index(tmpDbName());
  testDocRemoval(index);
}

TEST_F(OnDiskIndexTests, TestUpdate) {
  OnDiskIndex index(tmpDbName());
  testUpdate(index);
}

TEST_F(OnDiskIndexTests, SyntheticDataset) {
  OnDiskIndex index(tmpDbName());
  OnDiskIndex incremental_index(tmpDbName());

  size_t vocab_size = 10000;
  size_t n_docs = 1000;
  size_t topk = 10;

  auto [ids, docs, queries] = makeDocsAndQueries(vocab_size, n_docs);

  index.index(ids, docs);

  auto results = index.queryBatch(queries, /*k=*/topk);

  for (size_t i = 0; i < queries.size(); i++) {
    // i-th query goes to i-th doc.
    ASSERT_EQ(results[i][0].first, i);
    // Check single query vs batch query consistency.
    ASSERT_EQ(index.query(queries[i], /*k=*/topk, true), results[i]);
  }

  // Check that building index incrementally gets the same results.
  size_t n_chunks = 10;
  size_t chunksize = n_docs / n_chunks;
  for (int i = 0; i < n_chunks; i++) {
    size_t start = i * chunksize;
    size_t end = start + chunksize;
    incremental_index.index({ids.begin() + start, ids.begin() + end},
                            {docs.begin() + start, docs.begin() + end});
  }

  auto incremental_results = incremental_index.queryBatch(queries, /*k=*/topk);

  ASSERT_EQ(results.size(), incremental_results.size());
  for (size_t i = 0; i < results.size(); i++) {
    compareResults(results[i], incremental_results[i]);
  }
}

TEST_F(OnDiskIndexTests, SaveLoad) {
  OnDiskIndex index(tmpDbName());

  testSaveLoad(index, [](const std::string& path) {
    return OnDiskIndex::load(path, false);
  });
}

TEST_F(OnDiskIndexTests, ReadOnly) {
  OnDiskIndex index(tmpDbName());

  size_t vocab_size = 1000;
  size_t n_docs = 100;

  auto [ids, docs, queries] = makeDocsAndQueries(vocab_size, n_docs);

  index.index({ids.begin(), ids.begin() + n_docs / 2},
              {docs.begin(), docs.begin() + n_docs / 2});

  auto original_results = index.queryBatch(queries, /*k=*/5);

  std::string save_path = tmpDbName();

  index.save(save_path);

  auto read_write_index = OnDiskIndex::load(save_path, /*read_only=*/false);
  auto read_only_index = OnDiskIndex::load(save_path, /*read_only=*/true);

  ASSERT_EQ(original_results, read_write_index->queryBatch(queries, /*k=*/5));
  ASSERT_EQ(original_results, read_only_index->queryBatch(queries, /*k=*/5));
}

TEST_F(OnDiskIndexTests, ShardedRetrieverSaveLoad) {
  IndexConfig config;
  config.shard_size = 24;
  ShardedRetriever index(config, tmpDbName());

  testSaveLoad(index, [](const std::string& path) {
    return ShardedRetriever::load(path, false);
  });
}

}  // namespace thirdai::search::tests