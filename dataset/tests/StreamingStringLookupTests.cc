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

class StreamingStringLookupTests {
 public:
  static uint32_t criticalLookup(StreamingStringLookup& lookup, std::string& string) {
    return lookup.registerNewString(string);
  }
};

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

std::vector<uint32_t> getUidsInParallel(StreamingStringLookup& lookup, std::vector<std::string>& strings) {
  std::vector<uint32_t> uids(strings.size());
#pragma omp parallel for default(none) shared(strings, uids, lookup)
  for (uint32_t idx = 0; idx < strings.size(); idx++) {
    uids[idx] = lookup.lookup(strings[idx]);
  }
  return uids;
}

std::vector<uint32_t> getUidsInCriticalSectionOnly(StreamingStringLookup& lookup, std::vector<std::string>& strings) {
  std::vector<uint32_t> uids(strings.size());
#pragma omp parallel for default(none) shared(strings, uids, lookup)
  for (uint32_t idx = 0; idx < strings.size(); idx++) {
    uids[idx] = StreamingStringLookupTests::criticalLookup(lookup, strings[idx]);
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

// TODO(Geordie): Compare with registerNewString() since it's basically the all-in-critical-section version of the lookup method.

TEST(StreamingStringLookupTests, DoesNotBreak) {
  for (uint32_t trial = 0; trial < 100; trial++) {
    auto strings = generateRandomStrings(/* n_unique = */ 1000, /* repetitions = */ 1000, /* len = */ 10);
    StreamingStringLookup lookup(/* n_unique = */ 1200);
    auto uids = getUidsInParallel(lookup, strings);
    auto reverted_strings = backToStrings(lookup, uids);
    assertStringsEqual(strings, reverted_strings);
  }
}

TEST(StreamingStringLookupTests, LowOverheadWhenSingleThread) {}

TEST(StreamingStringLookupTests, MuchFasterWhenMultiThread) {
  auto parallel_start = std::chrono::high_resolution_clock::now();
  for (uint32_t trial = 0; trial < 10; trial++) {
    auto strings = generateRandomStrings(/* n_unique = */ 1000, /* repetitions = */ 1000, /* len = */ 10);
    StreamingStringLookup lookup(/* n_unique = */ 1200);
    auto uids = getUidsInParallel(lookup, strings);
  }
  auto parallel_end = std::chrono::high_resolution_clock::now();
  auto parallel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(parallel_end - parallel_start).count();

  auto critical_start = std::chrono::high_resolution_clock::now();
  for (uint32_t trial = 0; trial < 10; trial++) {
    auto strings = generateRandomStrings(/* n_unique = */ 1000, /* repetitions = */ 1000, /* len = */ 10);
    StreamingStringLookup lookup(/* n_unique = */ 1200);
    auto uids = getUidsInCriticalSectionOnly(lookup, strings);
  }
  auto critical_end = std::chrono::high_resolution_clock::now();
  auto critical_duration = std::chrono::duration_cast<std::chrono::milliseconds>(critical_end - critical_start).count();

  ASSERT_LE(parallel_duration, critical_duration);

  std::cout << "Parallel " << parallel_duration << "ms Critical " << critical_duration <<  "ms." << std::endl; 
}

} // namespace thirdai::dataset