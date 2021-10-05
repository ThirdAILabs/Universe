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
using thirdai::utils::MurmurHash;
using thirdai::utils::avalanche_testing::AvalancheTimedTestSuite;

uint64_t AvalancheTimedTestSuite::int_keys[num_keys];
std::string AvalancheTimedTestSuite::str_keys[num_keys];

/*
 * Tests speed of MurmurHash on integer and string keys.
 */
TEST_F(AvalancheTimedTestSuite, MurmurHashTimeTest) {
  // Allocate 64 bits for output of both keys.
  auto start =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch())
          .count();
  for (uint32_t i = 0; i < num_keys; i++) {
    MurmurHash(str_keys[i].c_str(),
               static_cast<uint32_t>(strlen(str_keys[i].c_str())), seed);
    MurmurHash(std::to_string(int_keys[i]).c_str(), sizeof(uint32_t), seed);
  }
  auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch())
                 .count();
  EXPECT_LE(end - start, 100);
}

/*
 * Tests Avalanche effect of MurmurHash on string keys.
 * https://crypto.stackexchange.com/questions/40268/hash-functions-and-the-avalanche-effect
 */
TEST_F(AvalancheTimedTestSuite, MurmurHashStringKeyAvalancheTest) {
  // Allocate 64 bits for both hash outputs.
  uint32_t murmurhash_output[2];
  uint32_t res[48][32] = {0};
  for (auto& str_key : str_keys) {
    murmurhash_output[0] = MurmurHash(
        str_key.c_str(), static_cast<uint32_t>(strlen(str_key.c_str())), seed);
    for (uint32_t j = 0; j < 48; j++) {
      std::bitset<48> str_key_flipped_bitarray(convert_to_bitstring(str_key));
      std::string str_key_flipped =
          str_key_flipped_bitarray.flip(j).to_string();
      murmurhash_output[1] = MurmurHash(
          str_key_flipped.c_str(),
          static_cast<uint32_t>(strlen(str_key_flipped.c_str())), seed);
      for (int k = 0; k < 32; k++) {
        res[j][k] += ((murmurhash_output[0] ^ murmurhash_output[1]) >> k) & 1;
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