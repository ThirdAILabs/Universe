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
  OnDiskIndexTests() {
    std::mt19937 process_rng(getpid());  // To handle parallel tests
    std::mt19937 time_rng(getpid());     // To handle parallel tests

    _prefix = "tmp_" + std::to_string(process_rng()) + "_" +
              std::to_string(time_rng()) + "_";
  }

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

TEST_F(OnDiskIndexTests, SyntheticDataset) {
  OnDiskIndex index(tmpDbName());
  OnDiskIndex incremental_index(tmpDbName());

  testSyntheticDataset(index, incremental_index);
}

TEST_F(OnDiskIndexTests, SaveLoad) {
  OnDiskIndex index(tmpDbName());

  testSaveLoad(index, [](const std::string& path) {
    return OnDiskIndex::load(path, false);
  });
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