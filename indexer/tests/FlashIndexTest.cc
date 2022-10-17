#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/DensifiedMinHash.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
#include <indexer/src/Flash.h>
#include <indexer/src/Indexer.h>
#include <indexer/tests/FlashIndexTestUtils.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using thirdai::automl::deployment::Flash;
using thirdai::hashing::DensifiedMinHash;

namespace thirdai::tests {

const uint32_t HASHES_PER_TABLE = 15;
const uint32_t NUM_TABLES = 100;
const uint32_t RANGE = 1000000;
const uint32_t NUM_VECTORS = 10000;

TEST(FlashIndexTest, SerializeAndDeserializeFlashIndexTest) {
  uint32_t input_vector_dimension = 100;

  uint32_t batch_size = 100;
  uint32_t num_queries = 100;
  uint32_t words_per_query = 10;
  uint32_t top_k = 5;

  auto random_vectors_for_index =
      createRandomSparseVectors(input_vector_dimension, NUM_VECTORS,
                                std::normal_distribution<float>(0, 1));

  auto batches = createBatches(random_vectors_for_index, batch_size);

  auto random_vectors_for_queries = createRandomSparseVectors(
      input_vector_dimension, 1000, std::normal_distribution<float>(0, 1));
  auto flash_index_queries =
      createBatches(random_vectors_for_queries, words_per_query);

  // Create a Flash Index
  auto* hash_function =
      new DensifiedMinHash(HASHES_PER_TABLE, NUM_TABLES, RANGE);
  auto flash_index = Flash<uint32_t>(hash_function);
  Flash<uint32_t> deserialized_index;

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

}  // namespace thirdai::tests