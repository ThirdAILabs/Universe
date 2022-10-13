#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/DensifiedMinHash.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
#include <indexer/src/Flash.h>
#include <indexer/src/Indexer.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

using thirdai::automl::deployment::Flash;
using thirdai::hashing::DensifiedMinHash;

namespace thirdai::testing {

const uint32_t HASHES_PER_TABLE = 15;
const uint32_t NUM_TABLES = 100;
const uint32_t RANGE = 1000000;
const uint32_t NUM_VECTORS = 10000;

/* Creates a vector of bolt batches for testing */
std::vector<BoltBatch> createBatches(std::vector<BoltVector>& input_vectors,
                                     uint32_t batch_size) {
  std::vector<BoltBatch> result;
  uint32_t current_vector_index = 0;
  while (current_vector_index < input_vectors.size()) {
    uint32_t next_batch_size = std::min(
        static_cast<uint32_t>(input_vectors.size() - current_vector_index),
        batch_size);

    std::vector<BoltVector> batch_vectors;

    for (uint32_t i = 0; i < next_batch_size; i++) {
      batch_vectors.push_back(
          std::move(input_vectors[current_vector_index + i]));
    }
    result.push_back(BoltBatch(std::move(batch_vectors)));
    current_vector_index += next_batch_size;
  }
  return result;
}

/* Generates random bolt vectors for testing */
std::vector<BoltVector> createRandomVectors(
    uint32_t dim, uint32_t num_vectors,
    std::normal_distribution<float> distribution) {
  std::vector<BoltVector> result;
  std::default_random_engine generator;
  for (uint32_t i = 0; i < num_vectors; i++) {
    BoltVector vec(/* l = */ dim, /* is_dense = */ true,
                   /* has_gradient = */ false);
    std::generate(vec.activations, vec.activations + dim,
                  [&]() { return distribution(generator); });
    result.push_back(std::move(vec));
  }
  return result;
}

TEST(FlashIndexTest, SerializeAndDeserializeFlashIndexTest) {
  uint32_t input_vector_dimension = 100;

  uint32_t batch_size = 100;
  uint32_t num_queries = 100;
  uint32_t words_per_query = 10;
  uint32_t top_k = 5;

  auto random_vectors_for_index =
      createRandomVectors(input_vector_dimension, NUM_VECTORS,
                          std::normal_distribution<float>(0, 1));

  auto batches = createBatches(random_vectors_for_index, batch_size);

  auto random_vectors_for_queries = createRandomVectors(
      input_vector_dimension, 1000, std::normal_distribution<float>(0, 1));
  auto flash_index_queries =
      createBatches(random_vectors_for_queries, words_per_query);

  // Create a Flash Index
  Flash<uint32_t> flash_index(
      DensifiedMinHash(HASHES_PER_TABLE, NUM_TABLES, RANGE));

  Flash<uint32_t> deserialized_index(
      DensifiedMinHash(HASHES_PER_TABLE, NUM_TABLES, RANGE));

  for (BoltBatch& batch : batches) {
    flash_index.addBatch(batch);
  }

  std::vector<std::vector<std::vector<uint32_t>>> query_outputs;
  for (BoltBatch& query : flash_index_queries) {
    auto output_vectors = flash_index.queryBatch(query, top_k, false);
    query_outputs.push_back(output_vectors);
  }

  // Serialize
  std::stringstream stream;
  {
    cereal::BinaryOutputArchive output_archive(stream);
    output_archive(flash_index);
  }

  // Deserialize
  {
    cereal::BinaryInputArchive input_archive(stream);
    input_archive(deserialized_index);
  }

  std::vector<std::vector<std::vector<uint32_t>>> second_query_outputs;
  for (BoltBatch& query : flash_index_queries) {
    auto output_vectors = deserialized_index.queryBatch(query, top_k, false);
    second_query_outputs.push_back(output_vectors);
  }

  for (uint32_t batch_index = 0; batch_index < num_queries; batch_index++) {
    for (uint32_t vec_index = 0; vec_index < query_outputs[batch_index].size();
         vec_index++) {
      ASSERT_TRUE(query_outputs[batch_index][vec_index] ==
                  second_query_outputs[batch_index][vec_index]);
    }
  }
}

}  // namespace thirdai::testing