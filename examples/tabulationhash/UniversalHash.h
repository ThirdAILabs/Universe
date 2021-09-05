#include <array>
#include <string>

namespace thirdai {
/*
 * Cheaper Hash functions, if you need 64 bit hash use murmurhash or xxhash
 */
class UniversalHash {
 private:
  uint32_t _seed;
  uint32_t T[8][256];

 public:
  explicit UniversalHash(uint32_t seed);
  // uint32_t gethash(uint8_t key);
  uint32_t gethash(std::string key);
  // uint32_t gethash(uint32_t key);
  uint32_t gethash(uint64_t key);
  void getBatchHash(const uint8_t* keys, const uint32_t* hashes,
                    uint32_t batchSize, uint32_t numHashes);
  void getBatchHash(const uint32_t* keys, const uint32_t* hashes,
                    uint32_t batchSize, uint32_t numHashes);
  void getBatchHash(const uint64_t* keys, const uint32_t* hashes,
                    uint32_t batchSize, uint32_t numHashes);
  void getBatchHash(const char* keys, const uint32_t* hashes,
                    uint32_t batchSize, uint32_t numHashes);
};

}  // namespace thirdai
