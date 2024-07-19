#include "InvertedIndexTestUtils.h"
#include "RetrieverTests.h"
#include <gtest/gtest.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/Tokenizer.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <vector>

namespace thirdai::search::tests {

InvertedIndex indexWithShardSize(size_t shard_size) {
  IndexConfig config;
  config.shard_size = shard_size;

  return InvertedIndex(config);
}

TEST(InvertedIndexTests, BasicRetrieval) {
  InvertedIndex index;
  testBasicRetrieval(index);
  ASSERT_EQ(index.nShards(), 1);
}

TEST(InvertedIndexTests, BasicRetrievalSharded) {
  InvertedIndex index = indexWithShardSize(3);
  testBasicRetrieval(index);
  ASSERT_EQ(index.nShards(), 4);
}

TEST(InvertedIndexTests, LessFrequentTokensScoreHigher) {
  InvertedIndex index;
  testLessFrequentTokensScoreHigher(index);
}

TEST(InvertedIndexTests, RepeatedTokensInDocs) {
  InvertedIndex index;
  testRepeatedTokensInDocs(index);
}

TEST(InvertedIndexTests, RepeatedTokensInQuery) {
  InvertedIndex index;
  testRepeatedTokensInQuery(index);
}

TEST(InvertedIndexTests, ShorterDocsScoreHigherWithSameTokens) {
  InvertedIndex index;
  testShorterDocsScoreHigherWithSameTokens(index);
}

TEST(InvertedIndexTests, DocRemoval) {
  InvertedIndex index;
  testDocRemoval(index);
}

TEST(InvertedIndexTests, TestUpdate) {
  InvertedIndex index = indexWithShardSize(2);

  index.index({1, 2, 3, 4}, {"a b c d 1 2", "5 6", "7 8", "f g h 3 4"});
  ASSERT_EQ(index.nShards(), 2);

  checkQuery(index, "a b c d e f g h", {1, 4});

  index.update({2, 4}, {"a h", "e f"});

  checkQuery(index, "a b c d e f g h", {4, 1, 2});
}

TEST(InvertedIndexTests, SyntheticDataset) {
  InvertedIndex index = indexWithShardSize(240);
  InvertedIndex incremental_index;

  testSyntheticDataset(index, incremental_index);

  ASSERT_GT(index.nShards(), 1);
}

TEST(InvertedIndexTests, ShardedVsUnsharded) {
  size_t vocab_size = 1000;
  size_t n_docs = 100;
  size_t topk = 10;

  auto [ids, docs, queries] = makeDocsAndQueries(vocab_size, n_docs);

  InvertedIndex unsharded_index;
  unsharded_index.index(ids, docs);
  ASSERT_EQ(unsharded_index.nShards(), 1);

  auto unsharded_results = unsharded_index.queryBatch(queries, /*k=*/topk);

  InvertedIndex sharded_index = indexWithShardSize(24);
  sharded_index.index(ids, docs);
  ASSERT_GT(sharded_index.nShards(), 1);

  auto sharded_results = sharded_index.queryBatch(queries, /*k=*/topk);

  ASSERT_EQ(unsharded_results.size(), sharded_results.size());
  for (size_t i = 0; i < unsharded_results.size(); i++) {
    compareResults(unsharded_results[i], sharded_results[i]);
  }
}

TEST(InvertedIndexTests, SaveLoad) {
  InvertedIndex index = indexWithShardSize(24);

  testSaveLoad(index, InvertedIndex::load);

  ASSERT_GT(index.nShards(), 1);
}

}  // namespace thirdai::search::tests