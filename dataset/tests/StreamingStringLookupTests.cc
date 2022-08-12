#include <dataset/src/encodings/categorical/StreamingStringLookup.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <algorithm>
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

std::pair<std::vector<uint32_t>, std::vector<std::string>> getUidsInParallel(StreamingStringLookup& lookup, std::vector<std::string>& strings) {
  std::vector<uint32_t> uids(strings.size());
  std::vector<std::string> explanations(strings.size());
#pragma omp parallel for default(none) shared(strings, uids, lookup, explanations)
  for (uint32_t idx = 0; idx < strings.size(); idx++) {
    auto res = lookup.lookup(strings[idx]);
    uids[idx] = res.first;
    explanations[idx] = res.second;
  }
  return {uids, explanations};
}

std::vector<std::string> backToStrings(StreamingStringLookup& lookup, std::vector<uint32_t>& uids) {
  std::vector<std::string> strings;
  strings.reserve(uids.size());
  for (auto uid : uids) {
    strings.push_back(lookup.originalString(uid));
  }

  return strings;
}

void assertStringsEqual(std::vector<std::string>& strings_1, std::vector<std::string>& strings_2, std::vector<uint32_t>& uids, StreamingStringLookup& lookup, std::vector<std::string>& explanations) {
  ASSERT_EQ(strings_1.size(), strings_2.size());
  for (uint32_t idx = 0; idx < strings_1.size(); idx++) {
    if (strings_1[idx] != strings_2[idx]) {
      std::cout << "Error with lookup with explanation: " << explanations[idx] << std::endl;
      std::cout << "Index: " << idx << std::endl;
      std::cout << "String 1: " << strings_1[idx] << std::endl;
      std::cout << "String 2: " << strings_2[idx] << std::endl;
      std::cout << "Uid: " << uids[idx] << std::endl;
      std::cout << "First occurrence of " << strings_1[idx] << " in source is ";
      for (uint32_t first_idx = 0; first_idx < strings_1.size(); first_idx++) {
        if (strings_1[first_idx] == strings_1[idx]) {
          std::cout << first_idx << std::endl;
          std::cout << "Explanation: " << explanations[first_idx] << std::endl;
          break;
        }
      }
      std::cout << "First occurrence of " << strings_2[idx] << " in source is ";
      for (uint32_t first_idx = 0; first_idx < strings_2.size(); first_idx++) {
        if (strings_1[first_idx] == strings_2[idx]) {
          std::cout << first_idx << std::endl;
          std::cout << "Explanation: " << explanations[first_idx] << std::endl;
          break;
        }
      }
      std::cout << "Previous occurrence of " << strings_1[idx] << " in source is ";
      for (int prev_idx = idx - 1; prev_idx >= 0; prev_idx--) {
        if (strings_1[prev_idx] == strings_1[idx]) {
          std::cout << prev_idx << std::endl;
          std::cout << "Explanation: " << explanations[prev_idx] << std::endl;
          break;
        }
      }
      std::stringstream ss;
      ss << "log" << idx << ".txt";
      std::ofstream out(ss.str());
      lookup.writeToFile(out);
      out.close();
    }
    // ASSERT_EQ(strings_1[idx], strings_2[idx]);
  }
}

// TODO(Geordie): Compare with registerNewString() since it's basically the all-in-critical-section version of the lookup method.

TEST(StreamingStringLookupTests, DoesNotBreak) {
  for (uint32_t trial = 0; trial < 100; trial++) {
    std::cout << "new trial" << std::endl;
    auto strings = generateRandomStrings(/* n_unique = */ 1000, /* repetitions = */ 1000, /* len = */ 10);
    StreamingStringLookup lookup(/* n_unique = */ 1000);
    auto [uids, explanations] = getUidsInParallel(lookup, strings);
    auto reverted_strings = backToStrings(lookup, uids);
    assertStringsEqual(strings, reverted_strings, uids, lookup, explanations);
  }
}

TEST(StreamingStringLookupTests, LowOverheadWhenSingleThread) {}

TEST(StreamingStringLookupTests, MuchFasterWhenMultiThread) {}

} // namespace thirdai::dataset