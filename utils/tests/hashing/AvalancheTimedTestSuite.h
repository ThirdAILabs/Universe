#include <gtest/gtest.h>
#include <bitset>
#include <iostream>
#include <random>

namespace thirdai::utils::avalanche_testing {
/*
 * Test Suite for generating random integer and string keys.
 * TODO(Alan or Josh): Decide how to organize this tests/hashing folder for
 * different kinds of tests.
 */
class AvalancheTimedTestSuite : public testing::Test {
 public:
  // Converts input string to bitstring (bitset).
  static std::string convert_to_bitstring(const std::string& str) {
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
      str_keys[i] = generate_random_string();
    }
  }

 private:
  static std::string generate_random_string() {
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
}  // namespace thirdai::utils::avalanche_testing