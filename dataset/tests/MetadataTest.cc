#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/metadata/Metadata.h>
#include <dataset/src/metadata/MetadataLoader.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::dataset {

TEST(MetadataTest, CorrectMetadataMapping) {
  uint32_t n_samples = 1000;

  std::unordered_map<std::string, uint32_t> key_vocab_map;
  key_vocab_map.reserve(n_samples);

  std::vector<BoltVector> vectors(1000);
  for (uint32_t i = 0; i < n_samples; i++) {
    key_vocab_map[std::to_string(i)] = i;
    vectors[i] = BoltVector::singleElementSparseVector(i);
  }

  auto metadata = Metadata::make(
      std::move(vectors),
      ThreadSafeVocabulary::make(std::move(key_vocab_map), /* fixed= */ true),
      /* dim= */ n_samples);

  auto metadata_key_vocab = metadata->getKeyToUidVocab();

  for (uint32_t i = 0; i < n_samples; i++) {
    auto key = std::to_string(i);
    auto vector = metadata->getVectorForKey(key);
    ASSERT_EQ(vector.len, 1);
    ASSERT_EQ(vector.activations[0], 1.0);
    ASSERT_EQ(vector.active_neurons[0], i);

    auto uid = metadata_key_vocab->getUid(key);
    auto vector_from_uid = metadata->getVectorForUid(uid);
    ASSERT_EQ(vector_from_uid.len, 1);
    ASSERT_EQ(vector_from_uid.activations[0], 1.0);
    ASSERT_EQ(vector_from_uid.active_neurons[0], i);
  }
}

TEST(MetadataTest, CorrectMetadataLoading) {
  std::string TEST_FILE = "test.csv";
  uint32_t n_samples = 1000;

  std::ofstream out(TEST_FILE);

  for (uint32_t i = 0; i < n_samples; i++) {
    out << i << std::endl;
  }

  out.close();

  auto data_loader = SimpleFileDataLoader::make(TEST_FILE,
                                                /* target_batch_size= */ 5);

  auto metadata = MetadataLoader::loadMetadata(
      data_loader, /* feature_blocks= */
      {NumericalCategoricalBlock::make(/* col= */ 0,
                                       /* n_classes= */ n_samples)},
      /* key_col= */ 0, /* n_unique_keys= */ n_samples);

  std::remove(TEST_FILE.c_str());

  for (uint32_t i = 0; i < n_samples; i++) {
    auto key = std::to_string(i);
    auto vector = metadata->getVectorForKey(key);
    ASSERT_EQ(vector.len, 1);
    ASSERT_EQ(vector.activations[0], 1.0);
    ASSERT_EQ(vector.active_neurons[0], i);
  }
}

}  // namespace thirdai::dataset