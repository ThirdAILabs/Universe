#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Graph.h>
#include <memory>
#include <optional>

namespace thirdai::data::tests {

State buildGraph() {
  auto graph = std::make_shared<automl::GraphInfo>(/* feature_dim= */ 4);

  State state(graph);

  ColumnMap columns({
      {"feature_1", ValueColumn<float>::make({0, 10, 20, 30})},
      {"feature_2", ValueColumn<float>::make({1, 11, 21, 31})},
      {"feature_3", ValueColumn<float>::make({2, 12, 22, 32})},
      {"feature_4", ValueColumn<float>::make({3, 13, 23, 33})},
      {"neighbors", ArrayColumn<uint32_t>::make(
                        {{1, 2, 3}, {4, 1}, {2}, {3, 2, 4, 1}}, std::nullopt)},
      {"id", ValueColumn<uint32_t>::make({1, 2, 3, 4}, std::nullopt)},
  });

  GraphBuilder builder("id", "neighbors",
                       {"feature_1", "feature_2", "feature_3", "feature_4"});

  builder.apply(columns, state);

  return state;
}

TEST(GraphTests, GraphBuilder) {
  auto state = buildGraph();

  const auto& graph = state.graph();

  ASSERT_EQ(graph->neighbors(1), std::vector<uint64_t>({1, 2, 3}));
  ASSERT_EQ(graph->neighbors(2), std::vector<uint64_t>({4, 1}));
  ASSERT_EQ(graph->neighbors(3), std::vector<uint64_t>({2}));
  ASSERT_EQ(graph->neighbors(4), std::vector<uint64_t>({3, 2, 4, 1}));

  ASSERT_EQ(graph->featureVector(1), std::vector<float>({0, 1, 2, 3}));
  ASSERT_EQ(graph->featureVector(2), std::vector<float>({10, 11, 12, 13}));
  ASSERT_EQ(graph->featureVector(3), std::vector<float>({20, 21, 22, 23}));
  ASSERT_EQ(graph->featureVector(4), std::vector<float>({30, 31, 32, 33}));
}

TEST(GraphTests, NeighborIds) {
  auto state = buildGraph();

  ColumnMap columns(
      {{"id", ValueColumn<uint32_t>::make({4, 3, 2, 1}, std::nullopt)}});

  NeighborIds neighbor_features("id", "nbrs");

  columns = neighbor_features.apply(columns, state);

  auto output = std::dynamic_pointer_cast<ArrayColumn<uint32_t>>(
      columns.getColumn("nbrs"));

  std::vector<std::vector<uint32_t>> expected_neighbors = {
      {3, 2, 4, 1}, {2}, {4, 1}, {1, 2, 3}};
  ASSERT_EQ(output->data(), expected_neighbors);
}

TEST(GraphTests, NeighborFeatures) {
  auto state = buildGraph();

  ColumnMap columns(
      {{"id", ValueColumn<uint32_t>::make({4, 3, 2, 1}, std::nullopt)}});

  NeighborFeatures neighbor_features("id", "normalized_features");

  columns = neighbor_features.apply(columns, state);

  auto output = std::dynamic_pointer_cast<ArrayColumn<float>>(
                    columns.getColumn("normalized_features"))
                    ->data();

  std::vector<std::vector<float>> expected_features = {
      {60.0 / 264, 64.0 / 264, 68.0 / 264, 72.0 / 264},
      {10.0 / 46, 11.0 / 46, 12.0 / 46, 13.0 / 46},
      {30.0 / 132, 32.0 / 132, 34.0 / 132, 36.0 / 132},
      {30.0 / 138, 33.0 / 138, 36.0 / 138, 39.0 / 138}};

  ASSERT_EQ(expected_features.size(), output.size());

  for (size_t row = 0; row < expected_features.size(); row++) {
    ASSERT_EQ(expected_features.at(row).size(), output.at(row).size());
    for (size_t col = 0; col < expected_features.at(row).size(); col++) {
      ASSERT_FLOAT_EQ(expected_features.at(row).at(col),
                      output.at(row).at(col));
    }
  }
}

}  // namespace thirdai::data::tests