#pragma once

#include <cstdint>

namespace thirdai::global_random {

uint32_t nextSeed();

void setBaseSeed(uint32_t seed);

}  // namespace thirdai::global_random