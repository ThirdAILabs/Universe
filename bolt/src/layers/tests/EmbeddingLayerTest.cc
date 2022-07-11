#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <hashing/src/MurmurHash.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt::tests {

constexpr uint32_t seed = 8274953;

// These tests are overriding the embedding block to be sequential integers, and
// then checking that the embeddings computed are sequential integers starting
// at the hash of the token.
class EmbeddingLayerTestFixture : public ::testing::Test {
 public:
  void SetUp() override {
    EmbeddingLayerConfig config(_num_lookups, _lookup_size, _log_block_size);
    _layer = std::make_unique<EmbeddingLayer>(config, seed);

    std::iota(_layer->_embedding_block.begin(), _layer->_embedding_block.end(),
              1.0);

    _layer->initializeLayer(/* new_batch_size= */ 4);
  }

  uint32_t getEmbeddingBlockSize() const {
    return _layer->_embedding_block_size_bytes;
  }

  uint64_t getHashLocFromLayer(uint32_t token, uint32_t lookup_index) const {
    return _layer->getEmbeddingBlockOffset(token, lookup_index);
  }

  uint32_t getHash(uint64_t token) const {
    return _layer->_hash_fn.gethash(token);
  }

  static std::vector<std::pair<uint64_t, uint64_t>> getDisjointRangesFromLayer(
      EmbeddingLayer& layer, std::vector<std::vector<uint64_t>>& hash_locs) {
    layer._embedding_block_offsets = std::move(hash_locs);

    return layer.getDisjointUpdateRanges();
  }

  float* getEmbeddingBlock() const { return _layer->_embedding_block.data(); }

  float* getEmbeddingGradients() const { return _layer->_gradients.data(); }

  uint32_t _lookup_size = 20, _num_lookups = 50, _log_block_size = 10;
  std::unique_ptr<EmbeddingLayer> _layer;
};

// Test that the hash locs computed are unique if the token or lookup_index
// changes.
TEST_F(EmbeddingLayerTestFixture, TestGetHashLoc) {
  ASSERT_NE(getHashLocFromLayer(/* token= */ 5, /* lookup_index= */ 17),
            getHashLocFromLayer(/* token= */ 17, /* lookup_index= */ 5));
  ASSERT_NE(getHashLocFromLayer(/* token= */ 5, /* lookup_index= */ 17),
            getHashLocFromLayer(/* token= */ 5, /* lookup_index= */ 18));
}

// Check that for a single token the embeddings contain the data at the correct
// index in the embedding block.
TEST_F(EmbeddingLayerTestFixture, SingleTokenEmbedding) {
  std::vector<uint32_t> tokens = {6, 18, 3};

  BoltBatch output = _layer->createBatchState(tokens.size());

  for (uint32_t i = 0; i < tokens.size(); i++) {
    _layer->forward(i, {tokens[i]}, output[i]);
  }

  for (uint32_t batch_index = 0; batch_index < tokens.size(); batch_index++) {
    const float* embedding = output[batch_index].activations;

    for (uint32_t lookup_index = 0; lookup_index < _num_lookups;
         lookup_index++) {
      uint64_t start = getHashLocFromLayer(tokens[batch_index], lookup_index);

      for (uint32_t j = 0; j < _lookup_size; j++) {
        ASSERT_EQ(embedding[lookup_index * _lookup_size + j], start + j + 1);
      }
    }
  }
}

// Check that with multiple tokens per input, the embedding is the sum of the
// contents of the embedding block at each hash location.
TEST_F(EmbeddingLayerTestFixture, MultipleTokenEmbedding) {
  std::vector<std::vector<uint32_t>> tokens = {
      {7, 4, 18}, {98, 34, 55, 2}, {9, 24}, {61, 75, 11}};

  BoltBatch output = _layer->createBatchState(tokens.size());

  for (uint32_t i = 0; i < tokens.size(); i++) {
    _layer->forward(i, tokens[i], output[i]);
  }

  for (uint32_t batch_index = 0; batch_index < tokens.size(); batch_index++) {
    const float* embedding = output[batch_index].activations;

    for (uint32_t lookup_index = 0; lookup_index < _num_lookups;
         lookup_index++) {
      for (uint32_t i = 0; i < _lookup_size; i++) {
        float expected_val = 0;
        for (uint32_t token : tokens[batch_index]) {
          uint64_t start = getHashLocFromLayer(token, lookup_index);

          expected_val += start + i + 1;
        }
        ASSERT_EQ(embedding[lookup_index * _lookup_size + i], expected_val);
      }
    }
  }
}

TEST_F(EmbeddingLayerTestFixture, Backpropagation) {
  std::vector<std::vector<uint32_t>> tokens = {
      {7, 4, 18}, {98, 34, 55, 2}, {9, 24}, {61, 75, 11}};

  BoltBatch output = _layer->createBatchState(tokens.size());

  for (uint32_t i = 0; i < tokens.size(); i++) {
    _layer->forward(i, tokens[i], output[i]);
  }

  std::unordered_map<uint32_t, float> deltas;

  for (uint32_t batch_index = 0; batch_index < tokens.size(); batch_index++) {
    for (uint32_t i = 0; i < _num_lookups * _lookup_size; i++) {
      // Make the gradient some semi-random value, make sure its a multiple of 2
      // to prevent floating point inpreceision.
      output[batch_index].gradients[i] = 0.5 * i + batch_index * 0.125;
    }

    _layer->backpropagate(batch_index, output[batch_index]);

    for (uint32_t token : tokens[batch_index]) {
      for (uint32_t lookup_index = 0; lookup_index < _num_lookups;
           lookup_index++) {
        uint64_t loc = getHashLocFromLayer(token, lookup_index);

        for (uint32_t i = 0; i < _lookup_size; i++) {
          deltas[loc + i] +=
              output[batch_index].gradients[lookup_index * _lookup_size + i];
        }
      }
    }
  }

  for (uint32_t i = 0; i < getEmbeddingBlockSize(); i++) {
    ASSERT_FLOAT_EQ(getEmbeddingGradients()[i], deltas[i]);
  }
}

// Test that the disjoint ranges are computed correctly for gradient updates,
// based off of the areas of the embedding block that are used.
TEST_F(EmbeddingLayerTestFixture, UpdateRangeCorrectness) {
  std::vector<std::vector<uint64_t>> test_hash_locs = {
      {4, 21, 68, 32, 99, 45, 2, 79}, {23, 82, 20, 32, 86, 63, 54, 47}};

  EmbeddingLayerConfig config(/* num_embedding_lookups= */ 4,
                              /* lookup_size= */ 5,
                              /* log_embedding_block_size= */ 7);
  EmbeddingLayer layer(config);

  std::vector<std::pair<uint64_t, uint64_t>> ranges =
      getDisjointRangesFromLayer(layer, test_hash_locs);

  std::vector<std::pair<uint64_t, uint64_t>> expected_ranges = {
      {2, 9},   {20, 28}, {32, 37}, {45, 52},
      {54, 59}, {63, 73}, {79, 91}, {99, 104}};

  ASSERT_EQ(ranges.size(), expected_ranges.size());

  for (uint32_t i = 0; i < ranges.size(); i++) {
    ASSERT_EQ(ranges.at(i), expected_ranges.at(i));
  }
}

}  // namespace thirdai::bolt::tests