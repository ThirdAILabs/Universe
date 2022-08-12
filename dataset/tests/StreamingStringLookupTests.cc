#include <dataset/src/encodings/categorical/StreamingStringLookup.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
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

std::vector<uint32_t> getUidsInParallel(StreamingStringLookup& lookup, std::vector<std::string>& strings) {
  std::vector<uint32_t> uids(strings.size());
#pragma omp parallel for default(none) shared(strings, uids, lookup)
  for (uint32_t idx = 0; idx < strings.size(); idx++) {
    uids[idx] = lookup.lookup(strings[idx]);
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
    StreamingStringLookup lookup(/* n_unique = */ 1000);
    auto uids = getUidsInParallel(lookup, strings);
    auto reverted_strings = backToStrings(lookup, uids);
    assertStringsEqual(strings, reverted_strings);
  }
}

TEST(StreamingStringLookupTests, LowOverheadWhenSingleThread) {}

TEST(StreamingStringLookupTests, MuchFasterWhenMultiThread) {}

} // namespace thirdai::dataset