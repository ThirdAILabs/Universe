#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/tests/BoltVectorTestUtils.h>
#include <hashing/src/DensifiedMinHash.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
#include <search/src/Flash.h>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

using thirdai::hashing::DensifiedMinHash;
using thirdai::search::Flash;

namespace thirdai::tests {

const uint32_t HASHES_PER_TABLE = 3;
const uint32_t NUM_TABLES = 32;
const uint32_t RANGE = 3000;
const uint32_t NUM_VECTORS = 100;

std::vector<uint32_t> createBatchLabels(uint32_t num_labels, uint32_t start) {
  std::vector<uint32_t> labels(num_labels);
  std::iota(labels.begin(), labels.end(), start);
  return labels;
}

TEST(FlashIndexTest, FlashIndexSerializationTest) {
  uint32_t input_vector_dimension = 50;

  uint32_t batch_size = 5;
  uint32_t num_queries = 12;
  uint32_t top_k = 5;

  auto vectors_to_be_indexed =
      createRandomSparseVectors(input_vector_dimension, NUM_VECTORS,
                                std::normal_distribution<float>(0, 1));

  auto flash_index_batches = createBatches(vectors_to_be_indexed, batch_size);

  auto query_vectors =
      createRandomSparseVectors(input_vector_dimension, num_queries,
                                std::normal_distribution<float>(0, 1));
  auto query_batches = createBatches(query_vectors, batch_size);

  // Create a Flash object
  auto flash_index = Flash(
      std::make_shared<DensifiedMinHash>(HASHES_PER_TABLE, NUM_TABLES, RANGE));

  uint32_t label_offset = 0;
  for (BoltBatch& batch : flash_index_batches) {
    label_offset += batch.getBatchSize();
    auto labels = createBatchLabels(batch.getBatchSize(), label_offset);
    flash_index.addBatch(batch, labels);
  }

  std::vector<std::vector<std::vector<uint32_t>>> query_outputs;

  for (BoltBatch& batch : query_batches) {
    auto [output_vectors, output_scores] =
        flash_index.queryBatch(batch, top_k, true);
    query_outputs.push_back(output_vectors);
  }

  // Serialization-Deserialization is not handled via load-save methods
  // because Flash is not a top level class.
  std::stringstream stream;
  {
    cereal::BinaryOutputArchive output_archive(stream);
    output_archive(flash_index);
  }

  Flash deserialized_flash_index;
  {
    cereal::BinaryInputArchive input_archive(stream);
    input_archive(deserialized_flash_index);
  }

  std::vector<std::vector<std::vector<uint32_t>>>
      deserialized_flash_query_outputs;

  for (BoltBatch& batch : query_batches) {
    auto [output_vectors, output_scores] =
        deserialized_flash_index.queryBatch(batch, top_k, true);
    deserialized_flash_query_outputs.push_back(output_vectors);
  }

  for (uint32_t batch_index = 0; batch_index < query_batches.size();
       batch_index++) {
    ASSERT_TRUE(query_outputs[batch_index] ==
                deserialized_flash_query_outputs[batch_index]);
  }
}
}  // namespace thirdai::tests