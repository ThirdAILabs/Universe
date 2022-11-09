#include <hashing/src/DWTA.h>
#include <gtest/gtest.h>
#include <cmath>

namespace thirdai::hashing {

// This test checks for a bug that causes hash values to exceed the expected
// hash range if the vector contains NaNs.
TEST(DWTATest, TestNoOutOfRangeHashWithNan) {
  std::vector<float> values = {NAN, NAN, NAN, NAN};

  uint32_t num_tables = 8, hashes_per_table = 4;
  DWTAHashFunction hash(
      /* input_dim= */ 4, /* _hashes_per_table= */ hashes_per_table,
      /* _num_tables= */ num_tables, /* range_pow= */ 3 * hashes_per_table,
      /* seed= */ 59302);

  std::vector<uint32_t> output_hashes(num_tables);
  hash.hashSingleDense(values.data(), values.size(), output_hashes.data());

  uint32_t range = 1 << (3 * hashes_per_table);
  for (uint32_t hash : output_hashes) {
    std::cout << hash << std::endl;
    ASSERT_LT(hash, range);
  }
}

}  // namespace thirdai::hashing