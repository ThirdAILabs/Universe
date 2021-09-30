#include <array>
#include <string>

namespace thirdai::utils {
/*
 * Cheaper Hash functions, if you need 64 bit hash use murmurhash or xxhash
 */
class UniversalHash {
 private:
  uint32_t _seed;
  uint32_t T[8][256];

 public:
  explicit UniversalHash(uint32_t seed);
  uint32_t gethash(const std::string& key);
  uint32_t gethash(uint64_t key);
};

}  // namespace thirdai::utils
