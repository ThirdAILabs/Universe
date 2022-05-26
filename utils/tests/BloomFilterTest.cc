#include <utils/src/BloomFilter.h>
#include <gtest/gtest.h>

TEST(BloomFilterTest, bitarrayfunction) {
    thirdai::utils::BloomFilter<uint32_t> bloom(100, 0.01);
}