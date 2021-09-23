#pragma once

#include <fstream>
#include <stdint.h>

struct SamplingConfig {
  uint32_t hashes_per_table, num_tables, range_pow, reservoir_size;

  SamplingConfig()
      : hashes_per_table(0), num_tables(0), range_pow(0), reservoir_size(0) {}

  SamplingConfig(uint32_t hashes_per_table, uint32_t num_tables,
                 uint32_t range_pow, uint32_t reservoir_size)
      : hashes_per_table(hashes_per_table),
        num_tables(num_tables),
        range_pow(range_pow),
        reservoir_size(reservoir_size) {}

  friend std::ostream& operator<<(std::ostream& out,
                                  const SamplingConfig& config);
};

