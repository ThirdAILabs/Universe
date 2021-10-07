#include "../../utils/hashing/MurmurHash.h"
#include "../src/EmbeddingLayer.h"
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt::tests {

class EmbeddingLayerTestFixture : public ::testing::Test {
 public:
  void SetUp() override {
    _layer = new EmbeddingLayer(_num_lookups, _lookup_size, _log_block_size,
                                /* seed for determinism */ 8274953);

    for (uint32_t i = 0; i < _layer->_embedding_block_size; i++) {
      _layer->_embedding_block[i] = i + 1;
    }
    _seed = _layer->_seed;
    _layer->SetBatchSize(4);
  }

  void TearDown() override { delete _layer; }

  uint32_t GetEmbeddingBlockSize() const { return _layer->_embedding_block_size; }

  const float* GetEmbeddingBlock() const { return _layer->_embedding_block; }

  uint32_t _lookup_size = 20, _num_lookups = 10, _log_block_size = 16, _seed;
  EmbeddingLayer* _layer;
};

TEST_F(EmbeddingLayerTestFixture, SingleTokenEmbedding) {
  std::vector<uint32_t> tokens = {6};

  for (uint32_t i = 0; i < tokens.size(); i++) {
    _layer->FeedForward(i, tokens.data() + i, 1);
  }

  for (uint32_t i = 0; i < tokens.size(); i++) {
    const float* embedding = _layer->GetEmbedding(i);

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
    _layer->FeedForward(i, tokens[i].data(), tokens[i].size());
  }

  for (uint32_t i = 0; i < tokens.size(); i++) {
    const float* embedding = _layer->GetEmbedding(i);

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
    _layer->FeedForward(i, tokens[i].data(), tokens[i].size());
  }

  std::unordered_map<uint32_t, float> deltas;

  for (uint32_t b = 0; b < 4; b++) {
    for (uint32_t i = 0; i < _num_lookups * _lookup_size; i++) {
      _layer->GetErrors(b)[i] = 0.5 * i + b * 0.005;
    }

    _layer->Backpropagate(b, 1.0);

    for (uint32_t t : tokens[b]) {
      for (uint32_t e = 0; e < _num_lookups; e++) {
        uint32_t id = t * _num_lookups + e;

        uint32_t loc = utils::MurmurHash(reinterpret_cast<const char*>(&id),
                                         sizeof(uint32_t), _seed);
        loc = loc >> (32 - _log_block_size);

        for (uint32_t j = 0; j < _lookup_size; j++) {
          deltas[loc + j] += _layer->GetErrors(b)[e * _lookup_size + j];
        }
      }
    }
  }

  for (uint32_t i = 0; i < GetEmbeddingBlockSize(); i++) {
    ASSERT_EQ(GetEmbeddingBlock()[i], i + 1 + deltas[i]);
  }
}

}  // namespace thirdai::bolt::tests