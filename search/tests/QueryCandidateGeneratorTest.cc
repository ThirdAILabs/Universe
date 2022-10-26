#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <search/src/Generator.h>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

using thirdai::bolt::QueryCandidateGeneratorConfig;

namespace thirdai::tests {

const uint32_t HASHES_PER_TABLE = 3;
const uint32_t NUM_TABLES = 32;
const uint32_t NUM_VECTORS = 100;

TEST(QueryCandidateGeneratorConfigTest,
     QueryCandidateGeneratorConfigSerializationTest) {
  const char* CONFIG_PATH = "./flash_config";
  auto config = QueryCandidateGeneratorConfig(
      /* hash_function = */ "DensifiedMinHash",
      /* num_tables = */ NUM_TABLES,
      /* hashes_per_table = */ HASHES_PER_TABLE,
      /* input_dim = */ NUM_VECTORS,
      /* top_k = */ 5,
      /* n_grams = */ {3, 4});

  config.save(/*config_file_name = */ CONFIG_PATH);

  auto deserialized_config =
      QueryCandidateGeneratorConfig::load(/* config_file_name = */ CONFIG_PATH);

  ASSERT_EQ(config, *deserialized_config);

  // Checks that config file was successfully removed
  EXPECT_EQ(std::remove(CONFIG_PATH), 0);
}

}  // namespace thirdai::tests