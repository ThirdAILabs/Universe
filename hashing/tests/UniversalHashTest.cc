#include "AvalancheTimedTestSuite.h"
#include <hashing/src/MurmurHash.h>
#include <hashing/src/UniversalHash.h>
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
using thirdai::hashing::AvalancheTimedTestSuite;
using thirdai::hashing::UniversalHash;

uint64_t AvalancheTimedTestSuite::int_keys[num_keys];
std::string AvalancheTimedTestSuite::str_keys[num_keys];
UniversalHash universal_hash(time(nullptr));

/*
 * Tests speed of UniversalHash on integer and string keys.
 */
TEST_F(AvalancheTimedTestSuite, UniversalHashTimeTest) {
  auto start =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch())
          .count();
  for (uint32_t i = 0; i < num_keys; i++) {
    universal_hash.gethash(str_keys[i]);
    universal_hash.gethash(int_keys[i]);
  }
  auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch())
                 .count();
  EXPECT_LE(end - start, 100);
}

/*
 * Tests Avalanche effect of UniversalHash on string keys.
 * https://crypto.stackexchange.com/questions/40268/hash-functions-and-the-avalanche-effect
 */
TEST_F(AvalancheTimedTestSuite, UniversalHashStringKeyAvalancheTest) {
  // Allocate 64 bits for both hash outputs.
  uint32_t tabulation_output[2];
  uint32_t output_bits_counter[48][32] = {0};
  for (auto& str_key : str_keys) {
    tabulation_output[0] = universal_hash.gethash(str_key);
    // Compute all possible 1 bit changes (48 bits in a string key input)
    for (uint32_t j = 0; j < 48; j++) {
      std::bitset<48> str_key_flipped_bitarray(convert_to_bitstring(str_key));
      std::string str_key_flipped =
          str_key_flipped_bitarray.flip(j).to_string();
      tabulation_output[1] = universal_hash.gethash(str_key_flipped);
      // Compute statistics of input and 32 bit output of each bit flip.
      for (int k = 0; k < 32; k++) {
        output_bits_counter[j][k] +=
            ((tabulation_output[0] ^ tabulation_output[1]) >> k) & 1;
      }
    }
  }

  // Expect ~0.5 probability of output bit changes for each input bit flip over
  // all 100000 keys.
  for (auto& output_bits_counter_per_input_flip : output_bits_counter) {
    for (auto& diff_count : output_bits_counter_per_input_flip) {
      EXPECT_NEAR(diff_count * 1.0 / num_keys, 0.5, 0.1);
    }
  }
}
