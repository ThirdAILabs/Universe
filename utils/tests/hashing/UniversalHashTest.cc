#include "../../hashing/MurmurHash.h"
#include "../../hashing/UniversalHash.h"
#include "AvalancheTimedTestSuite.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;
using thirdai::utils::UniversalHash;
using thirdai::utils::avalanche_testing::AvalancheTimedTestSuite;

uint64_t AvalancheTimedTestSuite::int_keys[num_keys];
std::string AvalancheTimedTestSuite::str_keys[num_keys];
UniversalHash universal_hash(time(nullptr));

TEST_F(AvalancheTimedTestSuite, UniversalHashTimeTest) {
  // Allocate 64 bits for output of both keys.
  uint32_t tabulation_output[2];
  // Test speed of Tabulation Hashing.
  auto start =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch())
          .count();
  for (uint32_t i = 0; i < num_keys; i++) {
    tabulation_output[0] = universal_hash.gethash(str_keys[i]);
    tabulation_output[1] = universal_hash.gethash(int_keys[i]);
  }
  auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch())
                 .count();
  std::cout << "Tabulation output ex: " << tabulation_output[0] << " "
            << tabulation_output[1] << std::endl;
  std::cout << "Tabulation Hash time (ms): " << end - start << std::endl;
  EXPECT_LE(end - start, 100);
}

TEST_F(AvalancheTimedTestSuite, UniversalHashStringKeyAvalancheTest) {
  // Allocate 64 bits for output of both keys.
  uint32_t tabulation_output[2];
  const uint32_t str_bitlength = 48;
  uint32_t res[str_bitlength][32] = {0};
  for (auto& str_key : str_keys) {
    tabulation_output[0] = universal_hash.gethash(str_key);
    for (uint32_t j = 0; j < str_bitlength; j++) {
      std::bitset<str_bitlength> str_key_flipped_bitarray(
          convert_to_bitstring(str_key));
      std::string str_key_flipped =
          str_key_flipped_bitarray.flip(j).to_string();
      tabulation_output[1] = universal_hash.gethash(str_key_flipped);
      for (int k = 0; k < 32; k++) {
        res[j][k] += ((tabulation_output[0] ^ tabulation_output[1]) >> k) & 1;
      }
    }
  }

  for (auto& j : res) {
    for (auto& k : j) {
      // Expect ~0.5 probability over all 100000 keys.
      EXPECT_NEAR(k, 50000, 10000);
    }
  }
}
