#include "UUID.h"

namespace thirdai::utils::uuid {

std::string getRandomHexString(uint32_t numBytesRandomness) {
  static std::random_device dev;
  static std::mt19937 rng(dev());

  std::string v = "0123456789ABCDEF";
  std::uniform_int_distribution<uint32_t> dist(0, v.size() - 1);

  std::string res;
  for (uint32_t i = 0; i < numBytesRandomness * 2; i++) {
    res += v.at(dist(rng));
  }
  return res;
}

// Since this uses randomness, it will be evaluated as a runtime constant, not
// a compile time constant, and will be different for every python interpreter
// session.
const std::string THIRDAI_UUID =
    getRandomHexString(/* numBytesRandomness = */ 16);
}  // namespace thirdai::utils::uuid
