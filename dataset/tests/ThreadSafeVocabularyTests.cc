#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/encodings/categorical/StringLookup.h>
#include <dataset/src/encodings/categorical/ThreadSafeVocabulary.h>
#include <sys/types.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
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

  std::vector<std::string> strings(n_unique * repetitions);
  for (uint32_t unique = 0; unique < n_unique; unique++) {
    std::string random_string;
    random_string.reserve(len);

    for (uint32_t i = 0; i < len; ++i) {
      random_string += alphanum[std::rand() % (sizeof(alphanum) - 1)];
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
                              std::vector<std::string>& strings) {
  std::vector<uint32_t> uids(strings.size());
  std::exception_ptr exception;
#pragma omp parallel for default(none) shared(strings, uids, lookup, exception)
  for (uint32_t idx = 0; idx < strings.size(); idx++) {
    try {
      uids[idx] = lookup.getUid(strings[idx]);
    } catch (...) {
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
    strings.push_back(*lookup.getString(uid));
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

template <typename LAMBDA_T>
uint32_t time(LAMBDA_T function) {
  auto start = std::chrono::high_resolution_clock::now();
  function();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

TEST(ThreadSafeVocabularyTests, Standalone) {
  uint32_t n_unique = 1000;
  auto strings =
      generateRandomStrings(n_unique, /* repetitions = */ 1000, /* len = */ 10);
  ThreadSafeVocabulary vocab(n_unique);
  auto uids = getUids(vocab, strings);
  auto reverted_strings = backToStrings(vocab, uids);
  assertStringsEqual(strings, reverted_strings);
}

TEST(ThreadSafeVocabularyTests, InBlock) {
  uint32_t n_classes = 1000;
  auto strings = generateRandomStrings(
      /* n_unique = */ n_classes, /* repetitions = */ 1000, /* len = */ 10);
  auto vocab = ThreadSafeVocabulary::make(n_classes);
  auto lookup_encoding = StringLookup::make(n_classes, vocab);
  auto lookup_block = std::make_shared<CategoricalBlock>(
      /* col = */ 0, /* encoding = */ lookup_encoding);

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
  auto lookup_encoding = StringLookup::make(n_unique, vocab);
  auto lookup_block_1 = std::make_shared<CategoricalBlock>(
      /* col = */ 0, /* encoding = */ lookup_encoding);
  auto lookup_block_2 = std::make_shared<CategoricalBlock>(
      /* col = */ 0, /* encoding = */ lookup_encoding);
  auto lookup_block_3 = std::make_shared<CategoricalBlock>(
      /* col = */ 0, /* encoding = */ lookup_encoding);

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

TEST(ThreadSafeVocabularyTests, SeenAllStringsBehavior) {
  auto seen_strings = generateRandomStrings(
      /* n_unique = */ 1000, /* repetitions = */ 1000, /* len = */ 10);
  ThreadSafeVocabulary vocab;

  auto before_seen_strings_duration =
      time([&]() { getUids(vocab, seen_strings); });

  vocab.fix();

  std::vector<uint32_t> uids;
  auto after_seen_strings_duration =
      time([&]() { uids = getUids(vocab, seen_strings); });

  ASSERT_LT(after_seen_strings_duration, before_seen_strings_duration);

  auto reverted_strings = backToStrings(vocab, uids);
  assertStringsEqual(seen_strings, reverted_strings);

  // Different string length so the strings are guaranteed to be unseen
  auto unseen_strings = generateRandomStrings(
      /* n_unique = */ 1, /* repetitions = */ 1, /* len = */ 20);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      vocab.getUid(unseen_strings.front()), std::invalid_argument);
}

}  // namespace thirdai::dataset