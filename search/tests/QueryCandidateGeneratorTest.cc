#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <exceptions/src/Exceptions.h>
#include <search/src/Generator.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

using thirdai::bolt::QueryCandidateGenerator;
using thirdai::bolt::QueryCandidateGeneratorConfig;

namespace thirdai::tests {

const uint32_t HASHES_PER_TABLE = 3;
const uint32_t NUM_TABLES = 32;
const uint32_t NUM_VECTORS = 100;

constexpr const char* CONFIG_FILE = "flash_config";
constexpr const char* QUERIES_FILE = "queries.csv";

const std::vector<std::string> INPUT_ROWS = {
    "Share my current location with Jim",
    "Does the View have reserved parking?",
    "What's happening this week at Smalls Jazz Club?",
    "How far am I from the Guggenheim museum?",
    "Show me the best museums to visit near my London Airbnb",
    "I need a cab to go to work",
    "Share my current location with Jim",
    "I need an Uber right now",
    "How far am I from the Guggenheim museum?",
    "Should I take a rain coat today?",
    "Share my current location with Jim",
    "I need a cab to go to work"};

void writeInputRowsToFile(const std::string& file_name,
                          const std::vector<std::string>& input_rows) {
  std::ofstream file(file_name);

  for (const auto& input_row : input_rows) {
    file << input_row << std::endl;
  }
}

QueryCandidateGeneratorConfig getQueryCandidateGeneratorConfig() {
  return QueryCandidateGeneratorConfig(
      /* hash_function = */ "DensifiedMinHash",
      /* num_tables = */ NUM_TABLES,
      /* hashes_per_table = */ HASHES_PER_TABLE,
      /* input_dim = */ NUM_VECTORS,
      /* top_k = */ 5,
      /* n_grams = */ {3, 4});
}

void assertQueryingWithoutTrainingThrowsException(
    QueryCandidateGenerator& query_candidate_generator) {
  ASSERT_THROW(query_candidate_generator.queryFromList( //NOLINT 
                   /* queries = */ {"first test query", "second test query"}),
               exceptions::QueryCandidateGeneratorException);
}

TEST(QueryCandidateGeneratorTest, QueryCandidateGeneratorConfigSerialization) {
  auto config = getQueryCandidateGeneratorConfig();

  config.save(/*config_file_name = */ CONFIG_FILE);

  auto deserialized_config =
      QueryCandidateGeneratorConfig::load(/* config_file_name = */ CONFIG_FILE);

  ASSERT_EQ(config, *deserialized_config);

  // Checks that config file was successfully removed
  EXPECT_EQ(std::remove(CONFIG_FILE), 0);
}

TEST(QueryCandidateGeneratorTest, GeneratorAssignUniqueLabels) {
  auto config = getQueryCandidateGeneratorConfig();

  auto query_candidate_generator = QueryCandidateGenerator::make(
      std::make_shared<QueryCandidateGeneratorConfig>(config));

  writeInputRowsToFile(QUERIES_FILE, INPUT_ROWS);

  assertQueryingWithoutTrainingThrowsException(query_candidate_generator);

  query_candidate_generator.buildFlashIndex(/* file_name = */ QUERIES_FILE);

  auto queries_to_labels_map =
      query_candidate_generator.getQueriesToLabelsMap();

  ASSERT_EQ(queries_to_labels_map.size(), INPUT_ROWS.size() - 4);

  EXPECT_EQ(std::remove(QUERIES_FILE), 0);
}

}  // namespace thirdai::tests