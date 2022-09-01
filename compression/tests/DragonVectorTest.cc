#include <gtest/gtest.h>
#include <compression/src/CompressedVector.h>
#include <compression/src/CompressionUtils.h>
#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>
namespace thirdai::compression::tests {

class DragonVectorTest : public testing::Test {
 private:
  std::mt19937 _rng;

 public:
  DragonVector<float> _vec;
  std::vector<float> _original_vec;
  uint32_t _uncompressed_size = 100;
  float _compression_density = 0.1;
  int _seed_for_hashing = 2;
  uint32_t sample_population_size = 50;

  DragonVectorTest() {
    std::uniform_int_distribution<int> dist(-200, 200);
    for (uint32_t i = 0; i < _uncompressed_size; i++) {
      _original_vec.push_back(static_cast<float>(dist(_rng)) / 64.0);
    }
    _vec = DragonVector<float>(_original_vec, _compression_density,
                               _seed_for_hashing, sample_population_size);
  }
};

// We have two constructors that takes in a vector or an array and compresses
// it. Testing those two constructors here
TEST_F(DragonVectorTest, ConstructorDragonVectorTest) {
  // we will now check whether the indices and the values are rightly set

  std::vector<uint32_t> indices_vec = _vec.indices();
  std::vector<float> values_vec = _vec.values();

  for (uint32_t i = 0; i < _vec.size(); i++) {
    if (indices_vec[i] != 0) {
      ASSERT_EQ(_original_vec[indices_vec[i]], values_vec[i]);
    }
  }
}

// Tests that atleast 50% of the spots in dragon vector are filled with indices
TEST_F(DragonVectorTest, EfficiencyDragonVectorTest) {
  int num_non_zeros = 0;
  for (auto indices : _vec.indices()) {
    if (indices != 0) {
      num_non_zeros++;
    }
  }
  ASSERT_GE(num_non_zeros, _vec.size() / 2);
}

// We are extending a dragon vector by itself. The original size of the dragon
// vector should remain the same.
TEST_F(DragonVectorTest, ExtendDragonVectorTest) {
  uint32_t size_before_extend = _vec.size();
  DragonVector<float> ns(_vec);
  _vec.extend(ns);

  std::vector<uint32_t> indices = _vec.indices();
  std::vector<float> values = _vec.values();

  for (uint32_t i = 0; i < size_before_extend; i++) {
    ASSERT_EQ(indices[i], indices[i + size_before_extend]);
    ASSERT_EQ(values[i], values[i + size_before_extend]);
  }
  ASSERT_EQ(_vec.uncompressedSize(), _uncompressed_size);
}

}  // namespace thirdai::compression::tests