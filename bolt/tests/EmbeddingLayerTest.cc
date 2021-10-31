#include "../../utils/hashing/MurmurHash.h"
#include "../layers/EmbeddingLayer.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt::tests {

class EmbeddingLayerTestFixture : public ::testing::Test {
 public:
  void SetUp() override {
    EmbeddingLayerConfig config(_num_lookups, _lookup_size, _log_block_size);
    _layer = new EmbeddingLayer(config,
                                /* seed for determinism */ 8274953);

    for (uint32_t i = 0; i < _layer->_embedding_block_size; i++) {
      _layer->_embedding_block[i] = i + 1;
    }
    _seed = _layer->_seed;
    _layer->initializeLayer(4);
  }

  void TearDown() override { delete _layer; }

  uint32_t getEmbeddingBlockSize() const {
    return _layer->_embedding_block_size;
  }

  std::vector<std::pair<uint64_t, uint64_t>> getDisjointRanges() {
    EmbeddingLayerConfig config(4, 5, 7);

    EmbeddingLayer layer(config);
    layer.initializeLayer(2);

    for (uint32_t i = 0; i < 2; i++) {
      layer._loc_lens[i] = 2;

      layer._embedding_locs[i] = new uint32_t[8];
      std::copy(dummyEmbeddingLocs[i].begin(), dummyEmbeddingLocs[i].end(),
                layer._embedding_locs[i]);
    }

    return layer.getDisjointUpdateRanges();
  }

  float* getEmbeddingBlock() const { return _layer->_embedding_block; }

  float* getEmbeddingGradients() const { return _layer->_gradients; }

  uint32_t _lookup_size = 20, _num_lookups = 50, _log_block_size = 10, _seed;
  EmbeddingLayer* _layer;

  std::vector<std::vector<uint32_t>> dummyEmbeddingLocs = {
      {4, 21, 68, 32, 99, 45, 2, 79}, {23, 82, 20, 32, 86, 63, 54, 47}};

  std::vector<std::pair<uint64_t, uint64_t>> expectedDisjointRanges = {
      {2, 9},   {20, 28}, {32, 37}, {45, 52},
      {54, 59}, {63, 73}, {79, 91}, {99, 104}};
};

TEST_F(EmbeddingLayerTestFixture, SingleTokenEmbedding) {
  std::vector<uint32_t> tokens = {6};

  for (uint32_t i = 0; i < tokens.size(); i++) {
    _layer->feedForward(i, tokens.data() + i, 1);
  }

  for (uint32_t i = 0; i < tokens.size(); i++) {
    const float* embedding = _layer->getEmbedding(i);

    for (uint32_t e = 0; e < _num_lookups; e++) {
      uint32_t item = tokens[i] * _num_lookups + e;
      uint32_t start = utils::MurmurHash(reinterpret_cast<const char*>(&item),
                                         sizeof(uint32_t), _seed);
      start = start >> (32 - _log_block_size);

      for (uint32_t j = 0; j < _lookup_size; j++) {
        ASSERT_EQ(embedding[e * _lookup_size + j], start + j + 1);
      }
    }
  }
}

TEST_F(EmbeddingLayerTestFixture, MultipleTokenEmbedding) {
  std::vector<std::vector<uint32_t>> tokens = {
      {7, 4, 18}, {98, 34, 55, 2}, {9, 24}, {61, 75, 11}};

  for (uint32_t i = 0; i < tokens.size(); i++) {
    _layer->feedForward(i, tokens[i].data(), tokens[i].size());
  }

  for (uint32_t i = 0; i < tokens.size(); i++) {
    const float* embedding = _layer->getEmbedding(i);

    for (uint32_t e = 0; e < _num_lookups; e++) {
      for (uint32_t j = 0; j < _lookup_size; j++) {
        float expected_val = 0;
        for (uint32_t t : tokens[i]) {
          uint32_t item = t * _num_lookups + e;
          uint32_t start = utils::MurmurHash(
              reinterpret_cast<const char*>(&item), sizeof(uint32_t), _seed);
          start = start >> (32 - _log_block_size);

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

  for (uint32_t i = 0; i < tokens.size(); i++) {
    _layer->feedForward(i, tokens[i].data(), tokens[i].size());
  }

  std::unordered_map<uint32_t, float> deltas;

  for (uint32_t b = 0; b < 4; b++) {
    for (uint32_t i = 0; i < _num_lookups * _lookup_size; i++) {
      _layer->getErrors(b)[i] = 0.5 * i + b * 0.005;
    }

    _layer->backpropagate(b);

    for (uint32_t t : tokens[b]) {
      for (uint32_t e = 0; e < _num_lookups; e++) {
        uint32_t id = t * _num_lookups + e;

        uint32_t loc = utils::MurmurHash(reinterpret_cast<const char*>(&id),
                                         sizeof(uint32_t), _seed);
        loc = loc >> (32 - _log_block_size);

        for (uint32_t j = 0; j < _lookup_size; j++) {
          deltas[loc + j] += _layer->getErrors(b)[e * _lookup_size + j];
        }
      }
    }
  }

  for (uint32_t i = 0; i < getEmbeddingBlockSize(); i++) {
    ASSERT_FLOAT_EQ(getEmbeddingGradients()[i], deltas[i]);
  }
}

TEST_F(EmbeddingLayerTestFixture, UpdateRangeCorrectness) {
  std::vector<std::pair<uint64_t, uint64_t>> ranges = getDisjointRanges();

  ASSERT_EQ(ranges.size(), expectedDisjointRanges.size());

  for (uint32_t i = 0; i < ranges.size(); i++) {
    ASSERT_EQ(ranges.at(i), expectedDisjointRanges.at(i));
  }
}

}  // namespace thirdai::bolt::tests