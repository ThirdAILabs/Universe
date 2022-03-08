#include <cereal/archives/binary.hpp>
#include <hashing/src/FastSRP.h>
#include <gtest/gtest.h>
#include <dataset/src/Vectors.h>
#include <flash/src/MaxFlashArray.h>
#include <memory>
#include <random>

using thirdai::dataset::DenseBatch;
using thirdai::dataset::DenseVector;

namespace thirdai::search {

/** Creates a vector of Batches with size batch_size that point to the
 * input_vectors */
//  TODO(Josh): Move to util
std::vector<DenseBatch> createBatches(std::vector<DenseVector>& input_vectors,
                                      uint32_t batch_size) {
  std::vector<DenseBatch> result;
  uint32_t current_vector_index = 0;
  while (current_vector_index < input_vectors.size()) {
    uint32_t next_batch_size = std::min(
        static_cast<uint32_t>(input_vectors.size() - current_vector_index),
        batch_size);

    std::vector<DenseVector> batch_vecs;
    for (uint32_t i = 0; i < next_batch_size; i++) {
      batch_vecs.push_back(
          std::move(input_vectors.at(current_vector_index + i)));
    }
    result.push_back(
        DenseBatch(std::move(batch_vecs), {}, current_vector_index));
    current_vector_index += next_batch_size;
  }
  return result;
}

std::vector<DenseVector> createRandomVectors(uint32_t dim,
                                             uint32_t num_vectors) {
  std::vector<DenseVector> result;
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0, 1);
  for (uint32_t i = 0; i < num_vectors; i++) {
    float* data = new float[dim];
    for (uint32_t d = 0; d < dim; d++) {
      data[d] = distribution(generator);
    }
    result.emplace_back(dim, data, true);
  }
  return result;
}

TEST(MaxFlashArrayTest, SerializeAndDeserializeTest) {
  uint32_t num_docs = 100;
  uint32_t words_per_doc = 100;
  uint32_t data_dim = 100;
  uint32_t words_per_query = 10;
  uint32_t num_queries = 100;

  auto doc_words = createRandomVectors(data_dim, num_docs * words_per_doc);
  auto documents = createBatches(doc_words, words_per_doc);

  auto query_words =
      createRandomVectors(data_dim, num_queries * words_per_query);
  auto queries = createBatches(query_words, words_per_query);

  // Create MaxFlashArray
  MaxFlashArray<uint8_t> to_serialize(
      new thirdai::hashing::FastSRP(100, 10, 10), 10, 100);
  MaxFlashArray<uint8_t> serialize_into;

  for (auto& d : documents) {
    to_serialize.addDocument(d);
  }

  std::vector<uint32_t> all_docs(num_docs);
  std::iota(all_docs.begin(), all_docs.end(), 0);
  std::vector<std::vector<float>> first_results;
  first_results.reserve(queries.size());
  for (auto& q : queries) {
    first_results.push_back(to_serialize.getDocumentScores(q, all_docs));
  }

  // Serialize
  std::stringstream ss;  // any stream can be used
  {
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(to_serialize);
  }  // archive goes out of scope, ensuring all contents are flushed

  // Deserialize
  {
    cereal::BinaryInputArchive iarchive(ss);  // Create an input archive
    iarchive(serialize_into);                 // Read the data from the archive
  }

  std::vector<std::vector<float>> second_results;
  second_results.reserve(queries.size());
  for (auto& q : queries) {
    second_results.push_back(serialize_into.getDocumentScores(q, all_docs));
  }

  for (uint32_t i = 0; i < num_queries; i++) {
    for (uint32_t j = 0; j < num_docs; j++) {
      ASSERT_EQ(first_results.at(i).at(j), second_results.at(i).at(j));
    }
  }
}

}  // namespace thirdai::search
