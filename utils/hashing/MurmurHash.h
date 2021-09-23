#pragma once

#include <cstdint>

extern "C" {
/**
 * Returns a murmur hash of `key' based on `seed'
 * using the MurmurHash3 algorithm
 */

uint32_t MurmurHash(const char* key, uint32_t len, uint32_t seed);
}