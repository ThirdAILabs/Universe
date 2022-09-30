#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <sys/types.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace thirdai::dataset {

static std::vector<std::string> generateRandomStrings(size_t n_unique,
                                                      size_t repetitions,
                                                      size_t len) {
  static const char alphanum[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";

  std::srand(0);

  std::vector<std::string> strings(n_unique * repetitions);
  for (uint32_t unique = 0; unique < n_unique; unique++) {
    std::string random_string;
    random_string.reserve(len);

    for (uint32_t i = 0; i < len; ++i) {
      random_string += alphanum[std::rand() % strlen(alphanum)];
    }

    for (uint32_t rep = 0; rep < repetitions; rep++) {
      strings[unique * repetitions + rep] = random_string;
    }
  }

  auto rng = std::default_random_engine{};
  std::shuffle(strings.begin(), strings.end(), rng);
  return strings;
}

std::vector<uint32_t> getUids(ThreadSafeVocabulary& lookup,
                              std::vector<std::string>& strings,
                              bool parallel=true) {
  std::vector<uint32_t> uids(strings.size());
  std::exception_ptr exception;
#pragma omp parallel for default(none) shared(strings, uids, lookup, exception) if(parallel)
  for (uint32_t idx = 0; idx < strings.size(); idx++) {
    try {
      uids[idx] = lookup.getUid(strings[idx]);
    } catch (...) {
#pragma omp critical
      exception = std::current_exception();
    }
  }
  if (exception) {
    std::rethrow_exception(exception);
  }
  return uids;
}

std::vector<std::string> backToStrings(ThreadSafeVocabulary& lookup,
                                       std::vector<uint32_t>& uids) {
  std::vector<std::string> strings;
  strings.reserve(uids.size());
  for (auto uid : uids) {
    strings.push_back(lookup.getString(uid));
  }

  return strings;
}

void assertStringsEqual(std::vector<std::string>& strings_1,
                        std::vector<std::string>& strings_2,
                        uint32_t exclude_last_n = 0) {
  ASSERT_EQ(strings_1.size(), strings_2.size());
  for (uint32_t idx = 0; idx < strings_1.size() - exclude_last_n; idx++) {
    ASSERT_EQ(strings_1[idx], strings_2[idx]);
  }
}

std::vector<uint32_t> getUidsFromBatch(BoltBatch& batch, uint32_t block_idx = 0,
                                       uint32_t block_dim = 0) {
  std::vector<uint32_t> uids;
  for (uint32_t i = 0; i < batch.getBatchSize(); i++) {
    uids.push_back(batch[i].active_neurons[block_idx] - block_idx * block_dim);
  }
  return uids;
}

TEST(ThreadSafeVocabularyTests, Standalone) {
  uint32_t n_unique = 100000;
  auto strings =
      generateRandomStrings(n_unique, /* repetitions = */ 20, /* len = */ 10);
  std::cout << "PARALLEL" << std::endl;
  {
    ThreadSafeVocabulary vocab(n_unique);
    for (uint32_t i = 0; i < 20; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      auto uids = getUids(vocab, strings);
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
    }
  }
  
  std::cout << "NOT PARALLEL" << std::endl;
  {
    ThreadSafeVocabulary vocab(n_unique);
    for (uint32_t i = 0; i < 20; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      auto uids = getUids(vocab, strings, false);
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
    }
  }
  // {
  //   ThreadSafeVocabulary vocab(n_unique);
  //   auto start = std::chrono::high_resolution_clock::now();
  //   auto uids = getUids(vocab, strings, false);
  //   auto end = std::chrono::high_resolution_clock::now();
  //   std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
  // }
  // {
  //   ThreadSafeVocabulary vocab(n_unique);
  //   auto start = std::chrono::high_resolution_clock::now();
  //   auto uids = getUids(vocab, strings, false);
  //   auto end = std::chrono::high_resolution_clock::now();
  //   std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
  // }
  // {
  //   ThreadSafeVocabulary vocab(n_unique);
  //   auto start = std::chrono::high_resolution_clock::now();
  //   auto uids = getUids(vocab, strings, false);
  //   auto end = std::chrono::high_resolution_clock::now();
  //   std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
  // }
  // {
  //   ThreadSafeVocabulary vocab(n_unique);
  //   auto start = std::chrono::high_resolution_clock::now();
  //   auto uids = getUids(vocab, strings, false);
  //   auto end = std::chrono::high_resolution_clock::now();
  //   std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
  // }
  
  ThreadSafeVocabulary vocab(n_unique);
  auto uids = getUids(vocab, strings);
  auto reverted_strings = backToStrings(vocab, uids);
  assertStringsEqual(strings, reverted_strings);
}

TEST(ThreadSafeVocabularyTests, InBlock) {
  uint32_t n_unique = 1000;
  auto strings =
      generateRandomStrings(n_unique, /* repetitions = */ 1000, /* len = */ 10);
  auto vocab = ThreadSafeVocabulary::make(n_unique);
  auto lookup_block = StringLookupCategoricalBlock::make(
      /* col = */ 0, vocab);
  
  GenericBatchProcessor processor(/* input_blocks = */ {lookup_block},
                                  /* label_blocks = */ {});
  auto [batch, _] = processor.createBatch(strings);

  auto uids = getUidsFromBatch(batch);
  auto reverted_strings = backToStrings(*vocab, uids);
  assertStringsEqual(strings, reverted_strings);
}

TEST(ThreadSafeVocabularyTests, InMultipleBlocks) {
  uint32_t n_unique = 1000;
  auto strings =
      generateRandomStrings(n_unique, /* repetitions = */ 1000, /* len = */ 10);
  auto vocab = ThreadSafeVocabulary::make(n_unique);
  auto lookup_block_1 = StringLookupCategoricalBlock::make(
      /* col = */ 0, vocab);
  auto lookup_block_2 = StringLookupCategoricalBlock::make(
      /* col = */ 0, vocab);
  auto lookup_block_3 = StringLookupCategoricalBlock::make(
      /* col = */ 0, vocab);

  GenericBatchProcessor processor(
      /* input_blocks = */ {lookup_block_1, lookup_block_2, lookup_block_3},
      /* label_blocks = */ {});
  auto [batch, _] = processor.createBatch(strings);

  uint32_t lookup_block_dim = lookup_block_1->featureDim();
  auto block_1_uids =
      getUidsFromBatch(batch, /* block_idx= */ 0, lookup_block_dim);
  auto block_2_uids =
      getUidsFromBatch(batch, /* block_idx= */ 1, lookup_block_dim);
  auto block_3_uids =
      getUidsFromBatch(batch, /* block_idx= */ 2, lookup_block_dim);

  auto block_1_reverted_strings = backToStrings(*vocab, block_1_uids);
  auto block_2_reverted_strings = backToStrings(*vocab, block_2_uids);
  auto block_3_reverted_strings = backToStrings(*vocab, block_3_uids);

  assertStringsEqual(strings, block_1_reverted_strings);
  assertStringsEqual(strings, block_2_reverted_strings);
  assertStringsEqual(strings, block_3_reverted_strings);
}

TEST(ThreadSafeVocabularyTests, UidOutOfRangeThrowsError) {
  std::unordered_map<std::string, uint32_t> string_to_uid_map = {
      {"string", 175}};

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      ThreadSafeVocabulary::make(std::move(string_to_uid_map),
                                 /* fixed = */ true),
      std::invalid_argument);
}

}  // namespace thirdai::dataset