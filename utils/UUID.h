#pragma once

#include <utils/Random.h>
#include <chrono>
#include <random>
#include <string>

namespace thirdai::utils::uuid {

class UUIDGenerator {
 public:
  explicit UUIDGenerator(uint32_t seed = global_random::nextSeed())
      : _rng(seed) {}

  uint64_t operator()() {
    uint32_t random_bytes = _rng();

    int64_t time = (std::chrono::system_clock::now().time_since_epoch() /
                    std::chrono::milliseconds(1));

    uint64_t uuid = 0;

    uuid |= (time & ((static_cast<uint64_t>(1) << 32) - 1));
    uuid |= (static_cast<uint64_t>(random_bytes) << 32);

    return uuid;
  }

 private:
  std::mt19937 _rng;
};

std::string getRandomHexString(uint32_t num_bytes_randomness);

// A unique identfier for this thirdai package instance. Currently used for
// licensing and telemetry.
extern const std::string THIRDAI_UUID;

}  // namespace thirdai::utils::uuid