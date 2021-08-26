#include "murmurhash/MurmurHash3.h"
#include "tabulationhash/UniversalHash.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

namespace ThirdAI {
class HashTest : public testing::Test {
 private:
  static std::string generate_random_string() {
    std::string str = "AAAAAA";
    str[0] = rand() % 26 + 65;
    str[1] = rand() % 26 + 65;
    str[2] = rand() % 26 + 65;

    str[3] = rand() % 10 + 48;
    str[4] = rand() % 10 + 48;
    str[5] = rand() % 10 + 48;
    return str;
  }

 protected:
  static uint64_t int_keys[100000];
  static std::string str_keys[100000];
  static const uint64_t seed = 1;
  static UniversalHash universal_hash;

  // Initialize all cross-test parameters (integer and string keys)
  static void SetUpTestSuite() {
    srand(seed);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> dis;
    // Generate 100000 random integer and char * keys
    for (int i = 0; i < 100000; i++) {
      int_keys[i] = dis(gen);
      str_keys[i] = generate_random_string();
    }
  }

 public:
  // Converts input string to bitstring (bitset).
  std::string convert_to_bitstring(std::string str) {
    std::string bitstring = "";
    for (char& _c : str) {
      bitstring += std::bitset<8>(_c).to_string();
    }
    return bitstring;
  }
};

uint64_t HashTest::int_keys[100000];
std::string HashTest::str_keys[100000];
UniversalHash HashTest::universal_hash(time(NULL));

TEST_F(HashTest, MurmurHashTimeTest) {
  // Allocate 64 bits for output of both keys.
  uint32_t murmurhash_output[2];
  // Test speed of MurmurHash.
  auto start =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch())
          .count();
  for (int i = 0; i < 100000; i++) {
    MurmurHash3_x86_32(str_keys[i].c_str(),
                       (uint64_t)strlen(str_keys[i].c_str()), seed,
                       &(murmurhash_output[0]));
    MurmurHash3_x86_32((void*)&(int_keys[i]), sizeof(uint64_t), seed,
                       &(murmurhash_output[1]));
  }
  auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch())
                 .count();
  std::cout << "MurmurHash output ex: " << murmurhash_output[0] << " "
            << murmurhash_output[1] << std::endl;
  std::cout << "MurmurHash time (ms): " << end - start << std::endl;
  EXPECT_LE(end - start, 100);
}

TEST_F(HashTest, TabulationHashTimeTest) {
  // Allocate 64 bits for output of both keys.
  uint32_t tabulation_output[2];
  // Test speed of Tabulation Hashing.
  auto start =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch())
          .count();
  for (int i = 0; i < 100000; i++) {
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

TEST_F(HashTest, MurmurHashStringKeyAvalancheTest) {
  // Allocate 64 bits for output of both keys.
  uint32_t murmurhash_output[2];
  uint32_t res[48][32] = {0};
  for (int i = 0; i < 100000; i++) {
    MurmurHash3_x86_32(str_keys[i].c_str(),
                       (uint64_t)strlen(str_keys[i].c_str()), seed,
                       &(murmurhash_output[0]));
    for (int j = 0; j < 48; j++) {
      std::bitset<48> str_key_flipped_bitarray(
          convert_to_bitstring(str_keys[i]));
      std::string str_key_flipped =
          str_key_flipped_bitarray.flip(j).to_string();
      MurmurHash3_x86_32(str_key_flipped.c_str(),
                         (uint64_t)strlen(str_key_flipped.c_str()), seed,
                         &(murmurhash_output[1]));
      for (int k = 0; k < 32; k++) {
        res[j][k] += ((murmurhash_output[0] ^ murmurhash_output[1]) >> k) & 1;
      }
    }
  }
  // TODO(alan): Could find better way for gtest assertion.
  for (int j = 0; j < 48; j++) {
    for (int k = 0; k < 32; k++) {
      // Expect ~0.5 probability over all 100000 keys.
      EXPECT_NEAR(res[j][k], 50000, 1000);
    }
  }
}

TEST_F(HashTest, MurmurHashIntegerKeyAvalancheTest) {
  // Allocate 64 bits for output of both keys.
  uint32_t murmurhash_output[2];
  uint32_t res[64][32] = {0};
  for (int i = 0; i < 100000; i++) {
    MurmurHash3_x86_32((void*)&(int_keys[i]), sizeof(uint64_t), seed,
                       &(murmurhash_output[1]));
    for (int j = 0; j < 64; j++) {
      uint64_t int_key_flipped = int_keys[i] ^ (1 << j);
      MurmurHash3_x86_32((void*)&(int_key_flipped), sizeof(uint64_t), seed,
                         &(murmurhash_output[1]));
      for (int k = 0; k < 32; k++) {
        res[j][k] += ((murmurhash_output[0] ^ murmurhash_output[1]) >> k) & 1;
      }
    }
  }
  // TODO(alan): Could find better way for gtest assertion.
  for (int j = 0; j < 64; j++) {
    for (int k = 0; k < 32; k++) {
      // Expect ~0.5 probability over all 100000 keys.
      EXPECT_NEAR(res[j][k], 50000, 20000);
    }
  }
}

TEST_F(HashTest, TabulationHashStringKeyAvalancheTest) {
  // Allocate 64 bits for output of both keys.
  uint32_t tabulation_output[2];
  uint32_t res[48][32] = {0};
  for (int i = 0; i < 100000; i++) {
    tabulation_output[0] = universal_hash.gethash(str_keys[i]);
    for (int j = 0; j < 48; j++) {
      std::bitset<48> str_key_flipped_bitarray(
          convert_to_bitstring(str_keys[i]));
      std::string str_key_flipped =
          str_key_flipped_bitarray.flip(j).to_string();
      tabulation_output[1] = universal_hash.gethash(str_key_flipped);
      for (int k = 0; k < 32; k++) {
        res[j][k] += ((tabulation_output[0] ^ tabulation_output[1]) >> k) & 1;
      }
    }
  }
  // TODO(alan): Could find better way for gtest assertion.
  for (int j = 0; j < 48; j++) {
    for (int k = 0; k < 32; k++) {
      // Expect ~0.5 probability over all 100000 keys.
      EXPECT_NEAR(res[j][k], 50000, 1000);
    }
  }
}

TEST_F(HashTest, TabulationHashIntegerKeyAvalancheTest) {
  // Allocate 64 bits for output of both keys.
  uint32_t tabulation_output[2];
  uint32_t res[64][32] = {0};
  for (int i = 0; i < 100000; i++) {
    tabulation_output[0] = universal_hash.gethash(int_keys[i]);
    for (int j = 0; j < 64; j++) {
      uint64_t int_key_flipped = int_keys[i] ^ (1 << j);
      tabulation_output[1] = universal_hash.gethash(int_key_flipped);
      for (int k = 0; k < 32; k++) {
        res[j][k] += ((tabulation_output[0] ^ tabulation_output[1]) >> k) & 1;
      }
    }
  }
  // TODO(alan): Could find better way for gtest assertion.
  for (int j = 0; j < 64; j++) {
    for (int k = 0; k < 32; k++) {
      // Expect ~0.5 probability over all 100000 keys.
      EXPECT_NEAR(res[j][k], 50000, 35000);
    }
  }
}
}  // namespace ThirdAI