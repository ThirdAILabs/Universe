#include <gtest/gtest.h>
#include "tabulationhash/UniversalHash.h"
#include "murmurhash/MurmurHash3.h"
#include <algorithm>
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

            // Initialize all cross-test parameters (integer and string keys)
            static void SetUpTestSuite() {
                srand(seed);
                // Generate 10000 random integer and char * keys
                for (int i = 0; i < 100000; i++) {
                    int_keys[i] = (uint64_t) rand() % 256;
                    str_keys[i] = generate_random_string();
                }
            }
    };

    uint64_t HashTest::int_keys[100000];
    std::string HashTest::str_keys[100000];

    TEST_F(HashTest, MurmurHashTimeTest) {
        // Allocate 64 bits for murmurhash output of both keys. TODO(alan): Reused for now. Avalanche testing later.
        uint32_t murmurhash_output[2];
        // Test speed of MurmurHash.
        auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        for (int i = 0; i < 100000; i++) {
            MurmurHash3_x86_32(str_keys[i].c_str(), (uint64_t)strlen(str_keys[i].c_str()), seed, &(murmurhash_output[0]));
            MurmurHash3_x86_32((void *)&(int_keys[i]), sizeof(uint64_t), seed, &(murmurhash_output[1]));
        }
        auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        std::cout << "MurmurHash output ex: " << murmurhash_output[0] << " " << murmurhash_output[1] << std::endl;
        std::cout << "MurmurHash time (ms): " << end - start << std::endl;
        EXPECT_LE(end - start, 20);
    }

    TEST_F(HashTest, TabulationHashTimeTest) {
        UniversalHash universal_hash(time(NULL));
        // Allocate 64 bits for tabulation hash output of both keys. TODO(alan): Reused for now. Avalanche testing later.
        uint32_t tabulation_output[2];
        // Test speed of Tabulation Hashing.
        auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        for (int i = 0; i < 100000; i++) {
            tabulation_output[0] = universal_hash.gethash(str_keys[i]);
            tabulation_output[1] = universal_hash.gethash(int_keys[i]);
        }
        auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        std::cout << "Tabulation output ex: " << tabulation_output[0] << " " << tabulation_output[1] << std::endl;
        std::cout << "Tabulation Hash time (ms): " << end - start << std::endl;
        EXPECT_LE(end - start, 20);
    }
}