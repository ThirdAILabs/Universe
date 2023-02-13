#include <gtest/gtest.h>
#include <auto_ml/src/dataset_factories/udt/GraphConfig.h>
#include <auto_ml/src/dataset_factories/udt/GraphDatasetFactory.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <googletest/googletest/include/gtest/gtest.h>
#include <memory>
#include <string>
namespace thirdai::automl::tests {

using data::GraphDatasetFactory;

const uint32_t num_nodes = 6;

const std::vector<std::vector<std::string>> rows = {
    {"0", "a", "b", "N", "1", "2", "3"}, {"1", "a", "c", "N", "2", "3", "4"},
    {"2", "c", "b", "N", "1", "2", "4"}, {"3", "c", "d", "Y", "2", "4", "1"},
    {"4", "b", "c", "Y", "2", "3", "4"}, {"5", "a", "d", "Y", "2", "5", "6"},
};

const std::vector<uint32_t> relationship_columns = {1, 2};

const std::vector<uint32_t> numerical_columns = {4, 5, 6};

uint32_t source_col = 0;

/*
The Graph that comes from this data is
0 -> 1,2,5
1 -> 0,4,5
2 -> 0,3
3 -> 2,5
4 -> 1
5 -> 0,1,3
*/

const std::vector<std::vector<std::string>> expected_adjacency_list = {
    {"1", "2", "5"}, {"0", "4", "5"}, {"0", "3"},
    {"2", "5"},      {"1"},           {"0", "1", "3"}};

const std::vector<std::vector<std::string>> expected_hop2_adjacency_list = {
    {"1", "2", "3", "4", "5"}, {"0", "2", "3", "4", "5"},
    {"0", "1", "3", "5"},      {"0", "1", "2", "5"},
    {"0", "1", "5"},           {"0", "1", "2", "3", "4"}};

TEST(GraphTest, correctGraphTest) {
  auto [adj_list, _] =
      GraphDatasetFactory::createGraph(rows, relationship_columns, source_col);

  for (const auto& node : adj_list) {
    uint32_t node_id = std::stoi(node.first);
    for (const auto& j : node.second) {
      ASSERT_TRUE(std::find(expected_adjacency_list[node_id].begin(),
                            expected_adjacency_list[node_id].end(),
                            j) != expected_adjacency_list[node_id].end());
    }
  }
}

TEST(GraphTest, correctNeighboursTest) {
  std::vector<std::string> nodes = {"0", "1", "2", "3", "4", "5"};
  dataset::ColumnNumberMap node_id_map(nodes);
  std::unordered_map<std::string, std::vector<std::string>> adjacency_list;
  for (uint32_t i = 0; i < num_nodes; i++) {
    adjacency_list[std::to_string(i)] = expected_adjacency_list[i];
  }
  auto hop1_neighbours = GraphDatasetFactory::findNeighboursForAllNodes(
      adjacency_list, 1, node_id_map);
  auto hop2_neighbours = GraphDatasetFactory::findNeighboursForAllNodes(
      adjacency_list, 2, node_id_map);
  auto hop3_neighbours = GraphDatasetFactory::findNeighboursForAllNodes(
      adjacency_list, 3, node_id_map);
  for (const auto& node_info : hop1_neighbours) {
    uint32_t node_id = std::stoi(node_info.first);
    for (const auto& k : node_info.second) {
      ASSERT_TRUE(std::find(expected_adjacency_list[node_id].begin(),
                            expected_adjacency_list[node_id].end(),
                            k) != expected_adjacency_list[node_id].end());
    }
  }

  for (const auto& node_info : hop2_neighbours) {
    uint32_t node_id = std::stoi(node_info.first);
    for (const auto& k : node_info.second) {
      ASSERT_TRUE(std::find(expected_hop2_adjacency_list[node_id].begin(),
                            expected_hop2_adjacency_list[node_id].end(),
                            k) != expected_hop2_adjacency_list[node_id].end());
    }
  }

  for (const auto& neighbour : hop3_neighbours) {
    ASSERT_EQ(neighbour.second.size(), 5);
  }
}

}  // namespace thirdai::automl::tests