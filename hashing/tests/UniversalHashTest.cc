#include <hashing/src/MurmurHash.h>
#include <hashing/src/UniversalHash.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

using thirdai::hashing::MurmurHash;
using thirdai::hashing::UniversalHash;

class UniversalHashTestSuite : public testing::Test {
 public:
  /*
   * Converts input string to bitstring (bitset).
   */
  static std::string convertToBitstring(const std::string& str) {
    std::string bitstring;
    for (const char& _c : str) {
      bitstring += std::bitset<8>(_c).to_string();
    }
    return bitstring;
  }

  const static uint32_t num_keys = 100000;
  static uint64_t int_keys[num_keys];
  static std::string str_keys[num_keys];
  static const uint64_t seed = 1;

  /*
   * Initialize all cross-test parameters (integer and string keys).
   */
  static void SetUpTestSuite() {
    srand(seed);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    // Generate 100000 random integer and char * keys
    for (uint32_t i = 0; i < num_keys; i++) {
      int_keys[i] = dis(gen);
      str_keys[i] = generateRandomString();
    }
  }

 private:
  /*
   * Generate random 6 character string.
   */
  static std::string generateRandomString() {
    const uint32_t num_chars = 26;
    const uint32_t starting_ascii = 65;
    std::string str = "AAAAAA";
    str[0] = rand() % num_chars + starting_ascii;
    str[1] = rand() % num_chars + starting_ascii;
    str[2] = rand() % num_chars + starting_ascii;
    str[3] = rand() % num_chars + starting_ascii;
    str[4] = rand() % num_chars + starting_ascii;
    str[5] = rand() % num_chars + starting_ascii;
    return str;
  }
};

uint64_t UniversalHashTestSuite::int_keys[num_keys];
std::string UniversalHashTestSuite::str_keys[num_keys];
UniversalHash universal_hash(time(nullptr));

/*
 * Tests Avalanche effect of UniversalHash on string keys.
 * https://crypto.stackexchange.com/questions/40268/hash-functions-and-the-avalanche-effect
 */
TEST_F(UniversalHashTestSuite, UniversalHashAvalancheTest) {
  // Allocate 64 bits for both hash outputs.
  uint32_t tabulation_output[2];
  uint32_t output_bits_counter[48][32] = {};
  for (auto& str_key : str_keys) {
    tabulation_output[0] = universal_hash.gethash(str_key);
    // Compute all possible 1 bit changes (48 bits in a string key input)
    for (uint32_t j = 0; j < 48; j++) {
      std::bitset<48> str_key_flipped_bitarray(convertToBitstring(str_key));
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

/*
 * Tests Avalanche effect of MurmurHash on string keys.
 * https://crypto.stackexchange.com/questions/40268/hash-functions-and-the-avalanche-effect
 */
TEST_F(UniversalHashTestSuite, MurmurHashAvalancheTest) {
  // Allocate 64 bits for both hash outputs.
  uint32_t murmurhash_output[2];
  uint32_t output_bits_counter[48][32] = {};
  for (auto& str_key : str_keys) {
    murmurhash_output[0] = MurmurHash(
        str_key.c_str(), static_cast<uint32_t>(strlen(str_key.c_str())), seed);
    // Compute all possible 1 bit changes (48 bits in a string key input)
    for (uint32_t j = 0; j < 48; j++) {
      std::bitset<48> str_key_flipped_bitarray(convertToBitstring(str_key));
      std::string str_key_flipped =
          str_key_flipped_bitarray.flip(j).to_string();
      murmurhash_output[1] = MurmurHash(
          str_key_flipped.c_str(),
          static_cast<uint32_t>(strlen(str_key_flipped.c_str())), seed);
      // Compute statistics of input and 32 bit output of each bit flip.
      for (int k = 0; k < 32; k++) {
        output_bits_counter[j][k] +=
            ((murmurhash_output[0] ^ murmurhash_output[1]) >> k) & 1;
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