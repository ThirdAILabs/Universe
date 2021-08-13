#include "murmurhash/MurmurHash3.h"
#include "tabulationhash/UniversalHash.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

std::string generate_random_string(int length)
{
    std::string possible_characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(0, possible_characters.size()-1);
    std::string ret = "";
    for(int i = 0; i < length; i++){
        int random_index = dist(engine); //get index between 0 and possible_characters.size()-1
        ret += possible_characters[random_index];
    }
    return ret;
}

int main(int argc, char** argv) {
  using namespace ThirdAI;
  UniversalHash universal_hash(time(NULL));

  // Generate 10000 random integer and char * keys
  uint64_t int_keys[100000];
  const char *str_keys[100000];
  uint64_t seed = 1;
  srand(seed);
  for (int i = 0; i < 100000; i++) {
    int_keys[i] = (uint64_t) rand() % 256;
    str_keys[i] = generate_random_string(100).c_str();
  }

  // Allocate 128 bits for hash output. TODO(alan): Reused for now. Avalanche testing later.
  uint64_t hash_otpt[2];

  // Test speed of MurmurHash.
  auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  for (int i = 0; i < 100000; i++) {
    MurmurHash3_x64_128(str_keys[i], (uint64_t)strlen(str_keys[i]), seed, hash_otpt);
    MurmurHash3_x64_128((void *)&(int_keys[i]), sizeof(uint64_t), seed, hash_otpt);
  }
  auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  std::cout << "MurmurHash time (ms): " << end - start << std::endl;
  
  // Test speed of Tabulation Hashing.
  start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  for (int i = 0; i < 100000; i++) {
    universal_hash.gethash(std::string(str_keys[i]));
    universal_hash.gethash(int_keys[i]);
  }
  end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  std::cout << "Tabulation Hash time (ms): " << end - start << std::endl;
  return 0;
}