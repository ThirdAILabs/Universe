#include <gtest/gtest.h>
#include <compression/src/DragonVector.h>
#include <sys/types.h>
#include <random>
#include <vector>
namespace thirdai::compression::tests {

class DragonVectorTest : public testing::Test {
 private:
  std::mt19937 _rng;

 public:
  DragonVector<float> _vec;
  DragonVector<float> _vec_from_array;

  std::vector<float> _original_vec;
  uint32_t _min_sketch_size = 10;
  uint32_t _original_size = 100;
  float _compression_density = 0.1;
  uint32_t _sketch_size =
      std::max(uint32_t(_compression_density * _original_size),
               std::min(uint32_t(_original_size), _min_sketch_size));

  int _seed_for_hashing = 5;

  DragonVectorTest() {
    std::uniform_int_distribution<int> dist(-100, 100);
    for (uint32_t i = 0; i < _original_size; i++) {
      _original_vec.push_back(static_cast<float>(dist(_rng)) / 64.0);
    }

    _vec = DragonVector<float>(_original_vec, _compression_density,
                               _seed_for_hashing);

    _vec_from_array =
        DragonVector<float>(_original_vec.data(), _compression_density,
                            _original_size, _seed_for_hashing);
  }
};

TEST_F(DragonVectorTest, ConstructorTest) {
  // size checks
  ASSERT_EQ(_vec.getOriginalSize(), _original_size);
  ASSERT_EQ(_vec_from_array.getOriginalSize(), _original_size);
  ASSERT_EQ(_vec.getSketchSize(), _sketch_size);
  ASSERT_EQ(_vec_from_array.getSketchSize(), _sketch_size);

  // we can probably remove these two asserts because compression density is not
  // integral to dragon vector
  ASSERT_EQ(_vec.getCompressionDensity(),
            _vec_from_array.getCompressionDensity());
  ASSERT_EQ(_vec.getCompressionDensity(), _compression_density);

  ASSERT_EQ(_vec.getSeedForHashing(), _seed_for_hashing);
  ASSERT_EQ(_vec_from_array.getSeedForHashing(), _seed_for_hashing);

  // we will now check whether the indices and the values are rightly set

  std::vector<uint32_t> indices_vec = _vec.getIndices();
  std::vector<float> values_vec = _vec.getValues();

  std::vector<uint32_t> indices_array = _vec_from_array.getIndices();
  std::vector<float> values_array = _vec_from_array.getValues();

  for (uint32_t i = 0; i < _sketch_size; i++) {
    if (indices_vec[i] != 0) {
      ASSERT_EQ(_original_vec[indices_vec[i]], values_vec[i]);
    }

    if (indices_array[i] != 0) {
      ASSERT_EQ(_original_vec[indices_array[i]], values_array[i]);
    }
  }
}

TEST_F(DragonVectorTest, GetSetTest) {
  std::vector<uint32_t> indices_vec = _vec.getIndices();

  for (uint32_t i = 0; i < _sketch_size; i++) {
    if (indices_vec[i] != 0) {
      ASSERT_EQ(_vec.get(indices_vec[i]), _original_vec[indices_vec[i]]);
      ASSERT_EQ(_vec[indices_vec[i]], _original_vec[indices_vec[i]]);
    }
  }

  std::mt19937 rng;
  std::uniform_int_distribution<uint32_t> dist(-100, 100);

  for (uint32_t i = 0; i < std::min(_sketch_size, uint32_t(10)); i++) {
    uint32_t index = uint32_t(std::abs(int(dist(rng) % _original_size)));
    float value = dist(rng) / 64.0;
    _vec.set(index, value);
    ASSERT_EQ(_vec.get(index), value);
  }
}

TEST_F(DragonVectorTest, ExtendTest) {
  DragonVector<float> ns(_vec);
  _vec.extend(ns);

  std::vector<uint32_t> indices = _vec.getIndices();
  std::vector<float> values = _vec.getValues();

  for (uint32_t i = 0; i < _sketch_size; i++) {
    ASSERT_EQ(indices[i], indices[i + _sketch_size]);
    ASSERT_EQ(values[i], values[i + _sketch_size]);
  }

  ASSERT_EQ(_vec.getOriginalSize(), _original_size);
}

TEST_F(DragonVectorTest, SplitTest) {
  size_t number_chunks = 3;
  std::vector<DragonVector<float>> splitVector = _vec.split(number_chunks);
  std::cout << "SplitVector has been made\n";
  uint32_t curr_vec = 0;
  uint32_t curr_index = 0;

  for (uint32_t i = 0; i < _sketch_size; i++, curr_index++) {
    if (curr_index == splitVector[curr_vec].getSketchSize()) {
      curr_vec++;
      curr_index = 0;
    }
    ASSERT_EQ(_vec.getIndices()[i],
              splitVector[curr_vec].getIndices()[curr_index]);
    ASSERT_EQ(_vec.getValues()[i],
              splitVector[curr_vec].getValues()[curr_index]);
  }
}

// add a test for addition of two dragon vectors
TEST_F(DragonVectorTest, AddTest) { (void)1; }

}  // namespace thirdai::compression::tests