#include <gtest/gtest.h>
#include <auto_ml/src/dataset_factories/udt/GraphConfig.h>
#include <auto_ml/src/dataset_factories/udt/GraphDatasetFactory.h>
#include <googletest/googletest/include/gtest/gtest.h>
#include <memory>
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

const std::vector<std::vector<uint32_t>> expected_adjacency_list = {
    {1, 2, 5}, {0, 4, 5}, {0, 3}, {2, 5}, {1}, {0, 1, 3}};

const std::vector<std::vector<uint32_t>> expected_hop2_adjacency_list = {
    {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 5},
    {0, 1, 2, 5},    {0, 1, 5},       {0, 1, 2, 3, 4}};

TEST(GraphTest, correctGraphTest) {
  auto adj_list = GraphDatasetFactory::createGraph(rows, relationship_columns);

  for (uint32_t i = 0; i < adj_list.size(); i++) {
    for (auto j : adj_list[i]) {
      ASSERT_TRUE(std::find(expected_adjacency_list[i].begin(),
                            expected_adjacency_list[i].end(),
                            j) != expected_adjacency_list[i].end());
    }
  }
}

TEST(GraphTest, correctNeighboursTest) {
  auto hop1_neighbours = GraphDatasetFactory::findNeighboursForAllNodes(
      num_nodes, expected_adjacency_list, 1);
  auto hop2_neighbours = GraphDatasetFactory::findNeighboursForAllNodes(
      num_nodes, expected_adjacency_list, 2);
  auto hop3_neighbours = GraphDatasetFactory::findNeighboursForAllNodes(
      num_nodes, expected_adjacency_list, 3);
  for (uint32_t i = 0; i < hop1_neighbours.size(); i++) {
    for (auto k : hop1_neighbours[i]) {
      ASSERT_TRUE(std::find(expected_adjacency_list[i].begin(),
                            expected_adjacency_list[i].end(),
                            k) != expected_adjacency_list[i].end());
    }
  }

  for (uint32_t i = 0; i < hop2_neighbours.size(); i++) {
    for (auto k : hop2_neighbours[i]) {
      ASSERT_TRUE(std::find(expected_hop2_adjacency_list[i].begin(),
                            expected_hop2_adjacency_list[i].end(),
                            k) != expected_hop2_adjacency_list[i].end());
    }
  }

  for (uint32_t i = 0; i < hop3_neighbours.size(); i++) {
    ASSERT_EQ(hop3_neighbours[i].size(), 5);
  }
}

TEST(GraphTest, numericaltest) {
  auto hop1_neighbours = GraphDatasetFactory::findNeighboursForAllNodes(
      num_nodes, expected_adjacency_list, 1);

  auto values = GraphDatasetFactory::processNumerical(rows, numerical_columns,
                                                      hop1_neighbours);

  for (uint32_t i = 0; i < values.size(); i++) {
    for (auto k : values[i]) {
      std::cout << k << " ";
    }
    std::cout << std::endl;
  }
}

}  // namespace thirdai::automl::tests