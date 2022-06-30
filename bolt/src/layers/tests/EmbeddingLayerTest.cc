#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <hashing/src/MurmurHash.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt::tests {

constexpr uint32_t seed = 8274953;

class EmbeddingLayerTestFixture : public ::testing::Test {
 public:
  void SetUp() override {
    EmbeddingLayerConfig config(_num_lookups, _lookup_size, _log_block_size);
    _layer = std::make_unique<EmbeddingLayer>(config, seed);

    for (uint32_t i = 0; i < _layer->_embedding_block_size; i++) {
      _layer->_embedding_block[i] = i + 1;
    }
    _layer->initializeLayer(4);
  }

  uint32_t getEmbeddingBlockSize() const {
    return _layer->_embedding_block_size;
  }

  uint64_t getHash(uint64_t id) const { return _layer->_hash_fn.gethash(id); }

  static std::vector<std::pair<uint64_t, uint64_t>> getDisjointRangesFromLayer(
      EmbeddingLayer& layer, std::vector<std::vector<uint64_t>>& hash_locs) {
    layer._hash_locs = std::move(hash_locs);

    return layer.getDisjointUpdateRanges();
  }

  float* getEmbeddingBlock() const { return _layer->_embedding_block.data(); }

  float* getEmbeddingGradients() const { return _layer->_gradients.data(); }

  uint32_t _lookup_size = 20, _num_lookups = 50, _log_block_size = 10;
  std::unique_ptr<EmbeddingLayer> _layer;
};

TEST_F(EmbeddingLayerTestFixture, SingleTokenEmbedding) {
  std::vector<uint32_t> tokens = {6};

  BoltBatch output = _layer->createBatchState(tokens.size());

  for (uint32_t i = 0; i < tokens.size(); i++) {
    _layer->forward(i, {tokens[i]}, output[i]);
  }

  for (uint32_t i = 0; i < tokens.size(); i++) {
    const float* embedding = output[i].activations;

    for (uint32_t e = 0; e < _num_lookups; e++) {
      uint64_t id = tokens[i] * _num_lookups + e;
      uint64_t start = getHash(id);
      start = start >> (64 - _log_block_size);

      for (uint32_t j = 0; j < _lookup_size; j++) {
        ASSERT_EQ(embedding[e * _lookup_size + j], start + j + 1);
      }
    }
  }
}

TEST_F(EmbeddingLayerTestFixture, MultipleTokenEmbedding) {
  std::vector<std::vector<uint32_t>> tokens = {
      {7, 4, 18}, {98, 34, 55, 2}, {9, 24}, {61, 75, 11}};

  BoltBatch output = _layer->createBatchState(tokens.size());

  for (uint32_t i = 0; i < tokens.size(); i++) {
    _layer->forward(i, tokens[i], output[i]);
  }

  for (uint32_t i = 0; i < tokens.size(); i++) {
    const float* embedding = output[i].activations;

    for (uint32_t e = 0; e < _num_lookups; e++) {
      for (uint32_t j = 0; j < _lookup_size; j++) {
        float expected_val = 0;
        for (uint32_t t : tokens[i]) {
          uint64_t id = t * _num_lookups + e;
          uint64_t start = getHash(id);
          start = start >> (64 - _log_block_size);

          expected_val += start + j + 1;
        }
        ASSERT_EQ(embedding[e * _lookup_size + j], expected_val);
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

  for (uint32_t b = 0; b < 4; b++) {
    for (uint32_t i = 0; i < _num_lookups * _lookup_size; i++) {
      output[b].gradients[i] = 0.5 * i + b * 0.125;
    }

    _layer->backpropagate(b, output[b]);

    for (uint32_t t : tokens[b]) {
      for (uint32_t e = 0; e < _num_lookups; e++) {
        uint64_t id = t * _num_lookups + e;
        uint64_t loc = getHash(id);
        loc = loc >> (64 - _log_block_size);

        for (uint32_t j = 0; j < _lookup_size; j++) {
          deltas[loc + j] += output[b].gradients[e * _lookup_size + j];
        }
      }
    }
  }

  for (uint32_t i = 0; i < getEmbeddingBlockSize(); i++) {
    ASSERT_FLOAT_EQ(getEmbeddingGradients()[i], deltas[i]);
  }
}

TEST_F(EmbeddingLayerTestFixture, UpdateRangeCorrectness) {
  std::vector<std::vector<uint64_t>> test_hash_locs = {
      {4, 21, 68, 32, 99, 45, 2, 79}, {23, 82, 20, 32, 86, 63, 54, 47}};

  EmbeddingLayerConfig config(4, 5, 7);
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