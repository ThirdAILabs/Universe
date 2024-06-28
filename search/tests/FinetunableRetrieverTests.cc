#include "InvertedIndexTestUtils.h"
#include <gtest/gtest.h>
#include <search/src/inverted_index/FinetunableRetriever.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <random>
#include <vector>

namespace thirdai::search::tests {

TEST(FinetunableRetrieverTests, Finetuning) {
  FinetunableRetriever retriever;

  retriever.index({1, 2, 3, 4}, {"a b c", "c d e", "e f g", "g h i"});
  checkQuery(retriever, "x y z b c", {1, 2});
  checkRank(retriever, "x y z b c", {1, 2, 3}, {1, 2});

  retriever.finetune({{2}, {3}, {4}}, {"x y z", "o p", "t q v"});

  checkQuery(retriever, "x y z b c", {2, 1});
  checkRank(retriever, "x y z b c", {1, 2, 3}, {2, 1});
}

TEST(FinetunableRetrieverTests, Associate) {
  FinetunableRetriever retriever;

  retriever.index({1, 2, 3, 4}, {"a b c", "c d e", "e f g", "g h i"});
  checkQuery(retriever, "x y z b c", {1, 2});
  checkRank(retriever, "x y z b c", {1, 2, 3}, {1, 2});

  retriever.associate({"x y z", "o p", "t q v"}, {"c d", "e f g", "g h i"},
                      /*strength=*/1);

  checkQuery(retriever, "x y z b c", {2, 1});
  checkRank(retriever, "x y z b c", {1, 2, 3}, {2, 1});
}

TEST(FinetunableRetrieverTests, RemoveDocs) {
  FinetunableRetriever retriever;

  retriever.index({1, 2, 3, 4}, {"a b c", "c d e", "e f g", "g h i"});
  checkQuery(retriever, "x y z b c", {1, 2});
  checkRank(retriever, "x y z b c", {1, 2, 3}, {1, 2});

  retriever.finetune({{2}, {4}, {4}}, {"x y z", "o p", "t q v"});

  checkQuery(retriever, "x y z b c", {2, 1});
  checkRank(retriever, "x y z b c", {1, 2, 3}, {2, 1});

  retriever.remove({2, 4});

  checkQuery(retriever, "x y z b c", {1});
  checkRank(retriever, "x y z b c", {1, 2, 3}, {1});
}

TEST(FinetunableRetrieverTests, SyntheticDataset) {
  size_t vocab_size = 10000;
  size_t n_docs = 1000;

  auto [ids, docs, queries] = makeDocsAndQueries(vocab_size, n_docs);

  FinetunableRetriever retriever;
  retriever.index(ids, docs);

  std::vector<std::vector<DocId>> finetuning_ids;
  finetuning_ids.reserve(ids.size());
  for (const auto& id : ids) {
    finetuning_ids.push_back({id});
  }
  retriever.finetune(finetuning_ids, queries);

  auto results = retriever.queryBatch(queries, /*k=*/5);

  for (size_t i = 0; i < queries.size(); i++) {
    // i-th query goes to i-th doc.
    ASSERT_EQ(results[i][0].first, i);
    // Check single query vs batch query consistency.
    ASSERT_EQ(retriever.query(queries[i], /*k=*/5), results[i]);
  }

  // Check that building index incrementally gets the same results.
  FinetunableRetriever incremental_retriever;
  size_t n_chunks = 10;
  size_t chunksize = n_docs / n_chunks;
  for (int i = 0; i < n_chunks; i++) {
    size_t start = i * chunksize;
    size_t end = start + chunksize;
    incremental_retriever.index({ids.begin() + start, ids.begin() + end},
                                {docs.begin() + start, docs.begin() + end});

    incremental_retriever.finetune(
        {finetuning_ids.begin() + start, finetuning_ids.begin() + end},
        {queries.begin() + start, queries.begin() + end});
  }

  auto incremental_results = incremental_retriever.queryBatch(queries, /*k=*/5);

  ASSERT_EQ(results, incremental_results);
}

TEST(FinetunableRetrieverTests, SaveLoad) {
  size_t vocab_size = 1000;
  size_t n_docs = 100;

  auto [ids, docs, queries] = makeDocsAndQueries(vocab_size, n_docs);

  FinetunableRetriever retriever;
  retriever.index({ids.begin(), ids.begin() + n_docs / 2},
                  {docs.begin(), docs.begin() + n_docs / 2});
  auto original_partial_results = retriever.queryBatch(queries, /*k=*/5);

  std::string save_path = "./finetunable_retriever.save";
  retriever.save(save_path);

  retriever.index({ids.begin() + n_docs / 2, ids.end()},
                  {docs.begin() + n_docs / 2, docs.end()});
  auto original_full_results = retriever.queryBatch(queries, /*k=*/5);

  auto loaded_retriever = FinetunableRetriever::load(save_path);

  auto loaded_partial_results = loaded_retriever->queryBatch(queries, /*k=*/5);

  ASSERT_EQ(original_partial_results, loaded_partial_results);

  loaded_retriever->index({ids.begin() + n_docs / 2, ids.end()},
                          {docs.begin() + n_docs / 2, docs.end()});
  auto loaded_full_results = loaded_retriever->queryBatch(queries, /*k=*/5);

  ASSERT_EQ(original_full_results, loaded_full_results);

  std::filesystem::remove_all(save_path);
}

}  // namespace thirdai::search::tests