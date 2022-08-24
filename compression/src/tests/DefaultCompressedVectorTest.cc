#include <gtest/gtest.h>
#include <compression/src/DefaultCompressedVector.h>
#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace thirdai::compression::tests {

class DefaultCompressedVectorTest : public testing::Test {
 private:
  std::mt19937 _rng;

 public:
  DefaultCompressedVector<float> _vec;
  int _num_samples = 50;

  std::vector<float> _original_vec;

  DefaultCompressedVectorTest() {
    std::uniform_real_distribution<float> dist(-100, 100);
    for (int i = 0; i < _num_samples; ++i) {
      _original_vec.push_back(dist(_rng));
    }
    _vec = DefaultCompressedVector<float>(_original_vec);
  }
};

TEST_F(DefaultCompressedVectorTest, ConstructorTest) {
  ASSERT_EQ(static_cast<uint32_t>(_num_samples), _vec.size());
  std::vector<float> values = _vec.getValues();
  for (int i = 0; i < _num_samples; ++i) {
    ASSERT_EQ(_original_vec[i], values[i]);
  }
}

TEST_F(DefaultCompressedVectorTest, GetSetTest) {
  std::mt19937 rng;
  std::vector<float> temp_vec;
  std::uniform_real_distribution<float> dist(-100, 100);
  temp_vec.reserve(_num_samples);
  for (int i = 0; i < _num_samples; ++i) {
    temp_vec.push_back(dist(rng));
  }

  for (int i = 0; i < _num_samples; i++) {
    ASSERT_EQ(_vec[i], _original_vec[i]);
    _vec.set(static_cast<uint32_t>(i), temp_vec[i]);
  }

  for (int i = 0; i < _num_samples; i++) {
    ASSERT_EQ(_vec[i], temp_vec[i]);
  }
}

}  // namespace thirdai::compression::tests