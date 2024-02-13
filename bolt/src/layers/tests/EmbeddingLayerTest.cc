#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/nn/optimizers/Adam.h>
#include <bolt_vector/src/BoltVector.h>
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
  static std::unique_ptr<EmbeddingLayer> createEmbeddingLayer(
      const std::string& reduction) {
    EmbeddingLayerConfig config(NUM_LOOKUPS, LOOKUP_SIZE, LOG_BLOCK_SIZE,
                                reduction, NUM_TOKENS_PER_INPUT);

    auto layer = std::make_unique<EmbeddingLayer>(config, seed);

    std::iota(layer->_embedding_block->begin(), layer->_embedding_block->end(),
              1.0);

    layer->initOptimizer(AdamFactory::make());

    return layer;
  }

  static uint32_t getEmbeddingBlockSize(
      std::unique_ptr<EmbeddingLayer>& layer) {
    return layer->_embedding_block_size;
  }

  static uint64_t getHashLocFromLayer(std::unique_ptr<EmbeddingLayer>& layer,
                                      uint32_t token, uint32_t lookup_index) {
    return layer->getEmbeddingBlockOffset(token, lookup_index);
  }

  static float* getEmbeddingBlock(std::unique_ptr<EmbeddingLayer>& layer) {
    return layer->_embedding_block->data();
  }

  static float* getEmbeddingGradients(std::unique_ptr<EmbeddingLayer>& layer) {
    return layer->_gradients.data();
  }

  static BoltBatch getEmbeddings(
      std::unique_ptr<EmbeddingLayer>& layer,
      const std::vector<std::vector<uint32_t>>& tokens) {
    BoltBatch output = layer->createBatchState(tokens.size());

    for (uint32_t i = 0; i < tokens.size(); i++) {
      layer->forward(
          BoltVector::makeSparseVector(
              tokens.at(i), std::vector<float>(tokens.at(i).size(), 1.0)),
          output[i]);
    }

    return output;
  }

  static void testEmbeddingBackpropagation(
      bool use_concat_reduction,
      const std::vector<std::vector<uint32_t>>& tokens) {
    auto layer = createEmbeddingLayer(use_concat_reduction ? "concat" : "sum");

    BoltBatch output = getEmbeddings(layer, tokens);

    std::unordered_map<uint32_t, float> gradients;

    for (uint32_t batch_index = 0; batch_index < tokens.size(); batch_index++) {
      for (uint32_t i = 0; i < output[batch_index].len; i++) {
        // Make the gradient some semi-random value, make sure its a multiple of
        // 2 to prevent floating point inpreceision.
        output[batch_index].gradients[i] = 0.5 * i + batch_index * 0.125;
      }

      layer->backpropagate(
          BoltVector::makeSparseVector(
              tokens.at(batch_index),
              std::vector<float>(tokens.at(batch_index).size(), 1.0)),
          output[batch_index]);

      /**
       * The location in the embedding block of a token's embedding originates
       * from is determined by the hash of the token. These loops look at the
       * gradients for each part of the output embedding, map it back to the
       * location in the embedding block it originated from, and sum up the
       * gradients for each location.
       */
      for (uint32_t token_idx = 0; token_idx < tokens[batch_index].size();
           token_idx++) {
        for (uint32_t lookup_index = 0; lookup_index < NUM_LOOKUPS;
             lookup_index++) {
          uint64_t loc = getHashLocFromLayer(
              layer, tokens[batch_index][token_idx], lookup_index);

          for (uint32_t i = 0; i < LOOKUP_SIZE; i++) {
            uint32_t gradient_offset;
            if (use_concat_reduction) {
              gradient_offset = token_idx * NUM_LOOKUPS * LOOKUP_SIZE +
                                lookup_index * LOOKUP_SIZE + i;
            } else {
              gradient_offset = lookup_index * LOOKUP_SIZE + i;
            }

            gradients[loc + i] +=
                output[batch_index].gradients[gradient_offset];
          }
        }
      }
    }

    for (uint32_t i = 0; i < getEmbeddingBlockSize(layer); i++) {
      ASSERT_FLOAT_EQ(getEmbeddingGradients(layer)[i], gradients[i]);
    }
  }

  static constexpr uint32_t LOOKUP_SIZE = 5, NUM_LOOKUPS = 8,
                            LOG_BLOCK_SIZE = 10, NUM_TOKENS_PER_INPUT = 3;

 private:
  static uint32_t getHash(std::unique_ptr<EmbeddingLayer>& layer,
                          uint64_t token) {
    return layer->_hash_fn.gethash(token);
  }
};

// Test that the hash locs computed are unique if the token or lookup_index
// changes.
TEST_F(EmbeddingLayerTestFixture, TestEmbeddingBlockOffsetUniqueness) {
  auto layer = createEmbeddingLayer("sum");

  ASSERT_NE(getHashLocFromLayer(layer, /* token= */ 5, /* lookup_index= */ 17),
            getHashLocFromLayer(layer, /* token= */ 17, /* lookup_index= */ 5));
  ASSERT_NE(getHashLocFromLayer(layer, /* token= */ 5, /* lookup_index= */ 17),
            getHashLocFromLayer(layer, /* token= */ 5, /* lookup_index= */ 18));
}

// Check that for a single token the embeddings with a sum reduction and a
// concatenation reduction are equivalent.
TEST_F(EmbeddingLayerTestFixture,
       SameOutputOfReductionsForSingleTokenEmbedding) {
  auto sum_embedding_layer = createEmbeddingLayer("sum");
  auto concat_embedding_layer = createEmbeddingLayer("concatenation");

  std::vector<std::vector<uint32_t>> tokens = {{6}, {18}, {3}};

  BoltBatch sum_output = getEmbeddings(sum_embedding_layer, tokens);
  BoltBatch concat_output = getEmbeddings(sum_embedding_layer, tokens);

  for (uint32_t batch_index = 0; batch_index < tokens.size(); batch_index++) {
    EXPECT_EQ(sum_output[batch_index].len, concat_output[batch_index].len);
    const float* sum_embedding = sum_output[batch_index].activations;
    const float* concat_embedding = concat_output[batch_index].activations;

    for (uint32_t i = 0; i < concat_output[batch_index].len; i++) {
      ASSERT_EQ(sum_embedding[i], concat_embedding[i]);
    }
  }
}

// Check that with multiple tokens per input, the embedding is the sum of the
// contents of the embedding block at each hash location.
TEST_F(EmbeddingLayerTestFixture, MultipleTokenEmbeddingSumReduction) {
  std::vector<std::vector<uint32_t>> tokens = {
      {7, 4, 18}, {98, 34, 55, 2}, {9, 24}, {61, 75, 11}};

  auto layer = createEmbeddingLayer("sum");

  BoltBatch output = getEmbeddings(layer, tokens);

  /**
   * Since the embedding block is a sequence of consecutive integers (done in
   * the createEmbeddingLayer helper function) we can check that the final
   * embedding at each index is the sum of the hashes of each token plus offset
   * within the embedding.
   */
  for (uint32_t batch_index = 0; batch_index < tokens.size(); batch_index++) {
    const float* embedding = output[batch_index].activations;

    for (uint32_t lookup_index = 0; lookup_index < NUM_LOOKUPS;
         lookup_index++) {
      for (uint32_t i = 0; i < LOOKUP_SIZE; i++) {
        float sum_of_token_embeddings_at_index = 0;
        for (uint32_t token : tokens[batch_index]) {
          uint64_t start = getHashLocFromLayer(layer, token, lookup_index);

          sum_of_token_embeddings_at_index += start + i + 1;
        }
        ASSERT_EQ(embedding[lookup_index * LOOKUP_SIZE + i],
                  sum_of_token_embeddings_at_index);
      }
    }
  }
}

// Check that with multiple tokens per input, the embedding is the concatenation
// of the contents of the embedding block at each hash location.
TEST_F(EmbeddingLayerTestFixture, MultipleTokenEmbeddingConcatReduction) {
  std::vector<std::vector<uint32_t>> tokens = {
      {7, 4, 18}, {98, 34, 55}, {9, 2, 24}, {61, 75, 11}};

  auto layer = createEmbeddingLayer("concatenation");

  BoltBatch output = getEmbeddings(layer, tokens);

  /**
   * Since the embedding block is a sequence of consecutive integers (done in
   * the createEmbeddingLayer helper function) we can check that the final
   * embedding foreach token is a sequence of integers starting at the hash of
   * the token.
   */
  for (uint32_t batch_index = 0; batch_index < tokens.size(); batch_index++) {
    ASSERT_EQ(output[batch_index].len,
              NUM_TOKENS_PER_INPUT * LOOKUP_SIZE * NUM_LOOKUPS);

    const float* embedding = output[batch_index].activations;

    uint32_t embedding_idx = 0;
    for (uint32_t token : tokens[batch_index]) {
      for (uint32_t lookup_index = 0; lookup_index < NUM_LOOKUPS;
           lookup_index++) {
        for (uint32_t i = 0; i < LOOKUP_SIZE; i++) {
          uint64_t start = getHashLocFromLayer(layer, token, lookup_index);
          float expected_val = start + i + 1;
          ASSERT_EQ(embedding[embedding_idx++], expected_val);
        }
      }
    }
  }
}

TEST_F(EmbeddingLayerTestFixture, BackpropagationSumReduction) {
  std::vector<std::vector<uint32_t>> tokens = {
      {7, 4, 18}, {98, 34, 55, 2}, {9, 24}, {61, 75, 11}};

  testEmbeddingBackpropagation(/* use_concat_reduction= */ false, tokens);
}

TEST_F(EmbeddingLayerTestFixture, BackpropagationConcatReduction) {
  std::vector<std::vector<uint32_t>> tokens = {
      {7, 4, 18}, {98, 34, 55}, {9, 2, 24}, {61, 75, 11}};

  testEmbeddingBackpropagation(/* use_concat_reduction= */ true, tokens);
}

}  // namespace thirdai::bolt::tests