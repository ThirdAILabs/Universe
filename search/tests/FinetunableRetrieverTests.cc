#include "InvertedIndexTestUtils.h"
#include <gtest/gtest.h>
#include <search/src/inverted_index/FinetunableRetriever.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <optional>
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

std::tuple<std::vector<std::vector<DocId>>, std::vector<std::string>,
           std::vector<std::string>>
makeFinetuningData(size_t original_vocab_size, size_t n_docs) {
  std::uniform_int_distribution<> query_len_dist(20, 70);

  // This function creates finetuning queries using a separate vocab from the
  // original dataset, so that the only way for it to get the correct answer is
  // with finetuning.
  Tokens vocab(original_vocab_size);
  for (size_t i = 0; i < original_vocab_size; i++) {
    vocab[i] = std::to_string(i + original_vocab_size);
  }

  std::vector<std::vector<DocId>> finetuning_ids;
  std::vector<std::string> finetuning_queries;
  std::vector<std::string> test_queries;

  std::mt19937 rng(7294);

  for (size_t i = 0; i < n_docs; i++) {
    Tokens query_tokens;
    std::sample(vocab.begin(), vocab.end(), std::back_inserter(query_tokens),
                query_len_dist(rng), rng);

    finetuning_ids.push_back({i});
    finetuning_queries.push_back(text::join(query_tokens, " "));

    Tokens test_query;
    int test_query_len = 0.8 * query_tokens.size();
    std::sample(query_tokens.begin(), query_tokens.end(),
                std::back_inserter(test_query), test_query_len, rng);
    test_queries.push_back(text::join(test_query, " "));
  }

  return {finetuning_ids, finetuning_queries, test_queries};
}

TEST(FinetunableRetrieverTests, SyntheticDataset) {
  size_t vocab_size = 10000;
  size_t n_docs = 1000;

  auto [ids, docs, unsup_queries] = makeDocsAndQueries(vocab_size, n_docs);

  FinetunableRetriever retriever;
  retriever.index(ids, docs);

  auto [finetuning_ids, finetuning_queries, sup_queries] =
      makeFinetuningData(vocab_size, n_docs);

  auto unsup_results = retriever.queryBatch(unsup_queries, /*k=*/5);
  for (size_t i = 0; i < unsup_queries.size(); i++) {
    // i-th query goes to i-th doc.
    ASSERT_EQ(unsup_results[i][0].first, i);
    // Check single query vs batch query consistency.
    ASSERT_EQ(retriever.query(unsup_queries[i], /*k=*/5), unsup_results[i]);
  }

  retriever.finetune(finetuning_ids, finetuning_queries);

  auto sup_results = retriever.queryBatch(sup_queries, /*k=*/5);
  for (size_t i = 0; i < unsup_queries.size(); i++) {
    // i-th query goes to i-th doc.
    ASSERT_EQ(sup_results[i][0].first, i);
    // Check single query vs batch query consistency.
    ASSERT_EQ(retriever.query(sup_queries[i], /*k=*/5), sup_results[i]);
  }

  // Check that building index incrementally gets the same results.
  FinetunableRetriever incr_retriever;
  const size_t n_chunks = 10;
  const size_t chunksize = n_docs / n_chunks;
  for (int start = 0; start < n_docs; start += chunksize) {
    incr_retriever.index(
        {ids.begin() + start, ids.begin() + start + chunksize},
        {docs.begin() + start, docs.begin() + start + chunksize});
  }

  auto unsup_incr_results = incr_retriever.queryBatch(unsup_queries, /*k=*/5);
  ASSERT_EQ(unsup_results, unsup_incr_results);

  for (int start = 0; start < n_docs; start += chunksize) {
    incr_retriever.finetune({finetuning_ids.begin() + start,
                             finetuning_ids.begin() + start + chunksize},
                            {finetuning_queries.begin() + start,
                             finetuning_queries.begin() + start + chunksize});
  }

  auto sup_incr_results = incr_retriever.queryBatch(sup_queries, /*k=*/5);
  ASSERT_EQ(sup_results, sup_incr_results);
}

void testFinetunableRetrieverSaveLoad(bool on_disk) {
  size_t vocab_size = 1000;
  size_t n_docs = 100;

  auto [ids, docs, unsup_queries] = makeDocsAndQueries(vocab_size, n_docs);

  auto [finetuning_ids, finetuning_queries, sup_queries] =
      makeFinetuningData(vocab_size, n_docs);

  std::optional<std::string> db_name = std::nullopt;
  if (on_disk) {
    db_name = randomPath() + ".db";
  }

  FinetunableRetriever retriever(IndexConfig(), db_name);

  retriever.index({ids.begin(), ids.begin() + n_docs / 2},
                  {docs.begin(), docs.begin() + n_docs / 2});
  retriever.finetune(
      {finetuning_ids.begin(), finetuning_ids.begin() + n_docs / 2},
      {finetuning_queries.begin(), finetuning_queries.begin() + n_docs / 2});

  auto original_partial_unsup = retriever.queryBatch(unsup_queries, /*k=*/5);
  auto original_partial_sup = retriever.queryBatch(sup_queries, /*k=*/5);

  std::string save_path = randomPath() + ".db";
  retriever.save(save_path);

  retriever.index({ids.begin() + n_docs / 2, ids.end()},
                  {docs.begin() + n_docs / 2, docs.end()});
  retriever.finetune(
      {finetuning_ids.begin() + n_docs / 2, finetuning_ids.end()},
      {finetuning_queries.begin() + n_docs / 2, finetuning_queries.end()});

  auto original_full_unsup = retriever.queryBatch(unsup_queries, /*k=*/5);
  auto original_full_sup = retriever.queryBatch(sup_queries, /*k=*/5);

  auto new_retriever =
      FinetunableRetriever::load(save_path, /*read_only=*/false);

  auto loaded_partial_unsup = new_retriever->queryBatch(unsup_queries, /*k=*/5);
  auto loaded_partial_sup = new_retriever->queryBatch(sup_queries, /*k=*/5);

  ASSERT_EQ(original_partial_unsup, loaded_partial_unsup);
  ASSERT_EQ(original_partial_sup, loaded_partial_sup);

  new_retriever->index({ids.begin() + n_docs / 2, ids.end()},
                       {docs.begin() + n_docs / 2, docs.end()});
  new_retriever->finetune(
      {finetuning_ids.begin() + n_docs / 2, finetuning_ids.end()},
      {finetuning_queries.begin() + n_docs / 2, finetuning_queries.end()});

  auto loaded_full_unsup = new_retriever->queryBatch(unsup_queries, /*k=*/5);
  auto loaded_full_sup = new_retriever->queryBatch(sup_queries, /*k=*/5);

  ASSERT_EQ(original_full_unsup, loaded_full_unsup);
  ASSERT_EQ(original_full_sup, loaded_full_sup);

  std::filesystem::remove_all(save_path);

  if (on_disk) {
    std::filesystem::remove_all(*db_name);
  }
}

TEST(FinetunableRetrieverTests, SaveLoadInMemory) {
  testFinetunableRetrieverSaveLoad(/*on_disk=*/false);
}

TEST(FinetunableRetrieverTests, SaveLoadOnDisk) {
  testFinetunableRetrieverSaveLoad(/*on_disk=*/true);
}

TEST(FinetunableRetrieverTests, SaveLoadReadOnly) {
  size_t vocab_size = 1000;
  size_t n_docs = 100;

  auto [ids, docs, unsup_queries] = makeDocsAndQueries(vocab_size, n_docs);

  auto [finetuning_ids, finetuning_queries, sup_queries] =
      makeFinetuningData(vocab_size, n_docs);

  std::string db_name = randomPath() + ".db";

  FinetunableRetriever retriever(IndexConfig(), db_name);

  retriever.index({ids.begin(), ids.end()}, {docs.begin(), docs.end()});
  retriever.finetune({finetuning_ids.begin(), finetuning_ids.end()},
                     {finetuning_queries.begin(), finetuning_queries.end()});

  std::string save_path = randomPath() + ".db";
  retriever.save(save_path);

  auto read_write = FinetunableRetriever::load(save_path, /*read_only=*/false);
  auto read_only = FinetunableRetriever::load(save_path, /*read_only=*/true);

  auto original_unsup = retriever.queryBatch(unsup_queries, /*k=*/5);
  auto original_sup = retriever.queryBatch(sup_queries, /*k=*/5);

  auto read_write_unsup = read_write->queryBatch(unsup_queries, /*k=*/5);
  auto read_write_sup = read_write->queryBatch(sup_queries, /*k=*/5);

  ASSERT_EQ(original_unsup, read_write_unsup);
  ASSERT_EQ(original_sup, read_write_sup);

  auto read_only_unsup = read_only->queryBatch(unsup_queries, /*k=*/5);
  auto read_only_sup = read_only->queryBatch(sup_queries, /*k=*/5);

  ASSERT_EQ(original_unsup, read_only_unsup);
  ASSERT_EQ(original_sup, read_only_sup);
}

}  // namespace thirdai::search::tests