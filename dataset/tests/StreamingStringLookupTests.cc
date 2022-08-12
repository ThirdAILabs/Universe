#include <dataset/src/encodings/categorical/StreamingStringLookup.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace thirdai::dataset {

static std::vector<std::string> generateRandomStrings(size_t n_unique, size_t repetitions, size_t len) {
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

  auto rng = std::default_random_engine {};
  std::shuffle(strings.begin(), strings.end(), rng);
  return strings;
}

std::vector<uint32_t> parallelGetUidsDefaultLookup(StreamingStringLookup& lookup, std::vector<std::string>& strings) {
  std::vector<uint32_t> uids(strings.size());
  std::cout << "Inside parallel" << std::endl;
#pragma omp parallel for default(none) shared(strings, uids, lookup)
  for (uint32_t idx = 0; idx < strings.size(); idx++) {
    uids[idx] = lookup.lookup(strings[idx]);
  }
  std::cout << "Outside parallel" << std::endl;
  return uids;
}

std::vector<uint32_t> parallelGetUidsCriticalLookup(StreamingStringLookup& lookup, std::vector<std::string>& strings) {
  std::vector<uint32_t> uids(strings.size());
#pragma omp parallel for default(none) shared(strings, uids, lookup)
  for (uint32_t idx = 0; idx < strings.size(); idx++) {
    uids[idx] = lookup.criticalLookup(strings[idx]);
  }
  return uids;
}

std::vector<uint32_t> sequentialGetUidsDefaultLookup(StreamingStringLookup& lookup, std::vector<std::string>& strings) {
  std::vector<uint32_t> uids(strings.size());
  for (uint32_t idx = 0; idx < strings.size(); idx++) {
    uids[idx] = lookup.lookup(strings[idx]);
  }
  return uids;
}

std::vector<uint32_t> sequentialGetUidsCriticalLookup(StreamingStringLookup& lookup, std::vector<std::string>& strings) {
  std::vector<uint32_t> uids(strings.size());
  for (uint32_t idx = 0; idx < strings.size(); idx++) {
    uids[idx] = lookup.criticalLookup(strings[idx]);
  }
  return uids;
}

std::vector<std::string> backToStrings(StreamingStringLookup& lookup, std::vector<uint32_t>& uids) {
  std::vector<std::string> strings;
  strings.reserve(uids.size());
  for (auto uid : uids) {
    strings.push_back(lookup.originalString(uid));
  }

  return strings;
}

void assertStringsEqual(std::vector<std::string>& strings_1, std::vector<std::string>& strings_2) {
  ASSERT_EQ(strings_1.size(), strings_2.size());
  for (uint32_t idx = 0; idx < strings_1.size(); idx++) {
    ASSERT_EQ(strings_1[idx], strings_2[idx]);
  }
}

template<typename LAMBDA_T>
auto time(LAMBDA_T lambda) {
  auto start = std::chrono::high_resolution_clock::now();
  lambda();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

TEST(StreamingStringLookupTests, DoesNotBreak) {
  for (uint32_t trial = 0; trial < 10000; trial++) {
    std::cout << "Trial " << trial << std::endl;
    auto strings = generateRandomStrings(/* n_unique = */ 1000, /* repetitions = */ 1000, /* len = */ 10);
    StreamingStringLookup lookup(/* n_unique = */ 1000);
    auto uids = parallelGetUidsDefaultLookup(lookup, strings);
    auto reverted_strings = backToStrings(lookup, uids);
    assertStringsEqual(strings, reverted_strings);
  }
}

TEST(StreamingStringLookupTests, LowOverheadWhenSingleThread) {
  auto strings = generateRandomStrings(/* n_unique = */ 1000, /* repetitions = */ 1000, /* len = */ 10);
  
  auto optimized_duration = time([&]() {
    for (uint32_t trial = 0; trial < 10; trial++) {
      StreamingStringLookup lookup(/* n_unique = */ 1000);
      auto uids = sequentialGetUidsDefaultLookup(lookup, strings);
    }
  });
  
  auto critical_duration = time([&]() {
    for (uint32_t trial = 0; trial < 10; trial++) {
      StreamingStringLookup lookup(/* n_unique = */ 1000);
      auto uids = sequentialGetUidsCriticalLookup(lookup, strings);
    }
  });
  
  ASSERT_LE(optimized_duration, 1.2 * critical_duration);

  std::cout << "Optimized " << optimized_duration << "ms vs Critical " << critical_duration <<  "ms." << std::endl; 
}

TEST(StreamingStringLookupTests, MuchFasterWhenMultiThread) {
  auto strings = generateRandomStrings(/* n_unique = */ 1000, /* repetitions = */ 1000, /* len = */ 10);
  
  auto optimized_duration = time([&]() {
    for (uint32_t trial = 0; trial < 10; trial++) {
      StreamingStringLookup lookup(/* n_unique = */ 1000);
      auto uids = parallelGetUidsDefaultLookup(lookup, strings);  
    }
  });
  
  auto critical_duration = time([&]() {
    for (uint32_t trial = 0; trial < 10; trial++) {
      StreamingStringLookup lookup(/* n_unique = */ 1000);
      auto uids = parallelGetUidsCriticalLookup(lookup, strings);
    }
  });
  
  ASSERT_LE(optimized_duration, critical_duration / 2);

  std::cout << "Optimized " << optimized_duration << "ms vs Critical " << critical_duration <<  "ms." << std::endl; 
}

} // namespace thirdai::dataset