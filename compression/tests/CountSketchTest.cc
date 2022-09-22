#include <gtest/gtest.h>
#include <compression/src/CountSketch.h>
#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using UniversalHash = thirdai::hashing::UniversalHash;

namespace thirdai::compression::tests {

class CountSketchTest : public testing::Test {
 private:
  std::mt19937 _rng;

 public:
  CountSketch<float> _vec;
  std::vector<float> _original_vec;
  uint32_t _uncompressed_size = 100;
  float _compression_density = 0.11;
  uint32_t _seed_for_sign = 1;
  uint32_t _seed_for_hashing_index = 0;

  CountSketchTest() {
    std::uniform_int_distribution<int> dist(-200, 200);
    for (uint32_t i = 0; i < _uncompressed_size; i++) {
      _original_vec.push_back(static_cast<float>(dist(_rng)) / 64.0);
    }

    _vec = CountSketch<float>(_original_vec, _compression_density, 1,
                              std::vector({_seed_for_hashing_index}),
                              std::vector({_seed_for_sign}));
  }
};

TEST_F(CountSketchTest, ConstructorCountSketchTest) {
  UniversalHash hash_for_index = UniversalHash(_seed_for_hashing_index);
  UniversalHash hash_for_sign = UniversalHash(_seed_for_sign);

  std::vector<float> count_sketch = _vec.sketches()[0];

  std::vector<float> test_count_sketch(
      (_uncompressed_size * _compression_density), 0);

  for (uint32_t i = 0; i < _uncompressed_size; i++) {
    uint32_t hashed_sign = hash_for_sign.gethash(i) % 2;
    uint32_t hashed_index =
        hash_for_index.gethash((i)) % test_count_sketch.size();

    if (hashed_sign == 0) {
      test_count_sketch[hashed_index] -= _original_vec[i];
    } else {
      test_count_sketch[hashed_index] += _original_vec[i];
    }
  }

  for (size_t i = 0; i < test_count_sketch.size(); i++) {
    ASSERT_EQ(test_count_sketch[i], count_sketch[i]);
  }
}

TEST_F(CountSketchTest, GetSetCountSketchTest) {
  UniversalHash hash_for_index = UniversalHash(_seed_for_hashing_index);
  UniversalHash hash_for_sign = UniversalHash(_seed_for_sign);

  std::vector<float> decompressed_vector(_uncompressed_size, 0);
  std::vector<float> count_sketch = _vec.sketches()[0];

  for (uint32_t i = 0; i < _uncompressed_size; i++) {
    uint32_t hashed_sign = hash_for_sign.gethash(i) % 2;
    uint32_t hashed_index = hash_for_index.gethash((i)) % count_sketch.size();
    if (hashed_sign == 0) {
      decompressed_vector[i] = -count_sketch[hashed_index];
    } else {
      decompressed_vector[i] = count_sketch[hashed_index];
    }
    ASSERT_EQ(decompressed_vector[i], _vec.get(i));
  }
}

TEST_F(CountSketchTest, ExtendCountSketchTest) {
  CountSketch<float> copy_sketch(_vec);
  std::vector<float> decompressed_vector = copy_sketch.decompress();

  _vec.extend(copy_sketch);

  /*
   * Since, we are extending a count sketch by it's copy, get function will just
   * return the average of the values in present in two indices which in this
   * case will be the same
   */
  for (uint32_t i = 0; i < _uncompressed_size; i++) {
    ASSERT_EQ(decompressed_vector[i], _vec.get(i));
  }
}

TEST_F(CountSketchTest, AddCountSketchTest) {
  CountSketch<float> copy_sketch(_vec);
  std::vector<float> decompressed_vector = copy_sketch.decompress();
  _vec.add(copy_sketch);

  /*
   * Adding a count sketch to itself is going to make all the stored values in
   * the sketches double and hence, the estimated values for an index will be
   * double of the initial.
   */
  for (uint32_t i = 0; i < _uncompressed_size; i++) {
    ASSERT_EQ(2 * decompressed_vector[i], _vec.get(i));
  }
}

TEST_F(CountSketchTest, SerializeCountSketchTest) {
  std::unique_ptr<char[]> serialized_data(new char[_vec.serialized_size()]);
  _vec.serialize(serialized_data.get());
  CountSketch<float> deserialized_vec(serialized_data.get());
  ASSERT_EQ(deserialized_vec.size(), _vec.size());
  ASSERT_EQ(deserialized_vec.uncompressedSize(), _vec.uncompressedSize());
  uint32_t num_sketches = deserialized_vec.numSketches();
  uint32_t sketch_size = deserialized_vec.size();

  std::vector<std::vector<float>> deserialized_sketches =
      deserialized_vec.sketches();
  std::vector<std::vector<float>> original_sketches = _vec.sketches();

  for (uint32_t num_sketch = 0; num_sketch < num_sketches; num_sketch++) {
    for (uint32_t index = 0; index < sketch_size; index++) {
      ASSERT_EQ(deserialized_sketches[num_sketch][index],
                original_sketches[num_sketch][index]);
    }
  }
}
}  // namespace thirdai::compression::tests