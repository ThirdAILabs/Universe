#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/DensifiedMinHash.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
#include <search/src/Flash.h>
#include <search/tests/FlashIndexTestUtils.h>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using thirdai::hashing::DensifiedMinHash;
using thirdai::search::Flash;

namespace thirdai::tests {

const uint32_t HASHES_PER_TABLE = 3;
const uint32_t NUM_TABLES = 32;
const uint32_t RANGE = 3000;
const uint32_t NUM_VECTORS = 100;

TEST(FlashIndexTest, FlashIndexSerializationTest) {
  uint32_t input_vector_dimension = 50;

  uint32_t batch_size = 20;
  uint32_t num_queries = 1;
  uint32_t words_per_query = 10;
  uint32_t top_k = 5;

  auto random_vectors_for_generator =
      createRandomSparseVectors(input_vector_dimension, NUM_VECTORS,
                                std::normal_distribution<float>(0, 1));

  auto batches = createBatches(random_vectors_for_generator, batch_size);

  auto random_vectors_for_queries =
      createRandomSparseVectors(input_vector_dimension, num_queries,
                                std::normal_distribution<float>(0, 1));
  auto queries = createBatches(random_vectors_for_queries, words_per_query);

  // Create a Flash object
  auto flash = Flash<uint32_t>(
      std::make_shared<DensifiedMinHash>(HASHES_PER_TABLE, NUM_TABLES, RANGE));
  Flash<uint32_t> deserialized_flash_instance;

  for (BoltBatch& batch : batches) {
    flash.addBatch(batch);
  }

  std::vector<std::vector<std::vector<uint32_t>>> query_outputs;
  for (BoltBatch& query : queries) {
    auto output_vectors = flash.queryBatch(query, top_k, false);
    query_outputs.push_back(output_vectors);
  }

  // Serialize
  std::stringstream stream;
  {
    cereal::BinaryOutputArchive output_archive(stream);
    output_archive(flash);
  }

  // Deserialize
  {
    cereal::BinaryInputArchive input_archive(stream);
    input_archive(deserialized_flash_instance);
  }

  std::vector<std::vector<std::vector<uint32_t>>> second_query_outputs;
  for (BoltBatch& query : queries) {
    auto output_vectors =
        deserialized_flash_instance.queryBatch(query, top_k, false);
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