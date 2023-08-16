#include "Random.h"
#include <ctime>
#include <random>

namespace thirdai::global_random {

namespace {

std::mt19937 rng{static_cast<uint32_t>(time(nullptr))};

}  // namespace

uint32_t nextSeed() { return rng(); }

void setBaseSeed(uint32_t seed) { rng = std::mt19937(seed); }

}  // namespace thirdai::global_random