#include <gtest/gtest.h>
#include "BlockTest.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/CategoricalTracking.h>
#include <dataset/src/encodings/categorical/StringToUidMap.h>
#include <dataset/src/encodings/categorical_history/CategoricalHistoryIndex.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <vector>

namespace thirdai::dataset {

class CategoricalTrackingBlockTests : public BlockTest {};

TEST_F(CategoricalTrackingBlockTests, CorrectOutputNoGraph) {
  StringMatrix mock_data {
    {"id_1", "2022-01-01", "0"},
    {"id_1", "2022-01-01", "1"},
    {"id_1", "2022-01-02", "2"},
    {"id_2", "2022-01-04", "3"},
    {"id_1", "2022-01-07", "4"},
  };

  std::vector<SegmentedSparseFeatureVector> vecs;
  auto id_map = std::make_shared<StringToUidMap>(3);
  auto index = std::make_shared<CategoricalHistoryIndex>(/* n_ids = */ 3, /* n_categories = */ 5, /* buffer_size = */ 5);
  CategoricalTrackingBlock block(/* id_col = */ 0, /* timestamp_col = */ 1, /* category_col = */ 2, /* horizon = */ 0, /* lookback = */ 2, id_map, index);

  for (auto& row : mock_data) {
    SegmentedSparseFeatureVector vec;
    addVectorSegmentWithBlock(block, row, vec);
    for (const auto [idx, _] : vectorEntries(vec)) {
      ASSERT_LT(idx, 5);
    }
    vecs.push_back(std::move(vec));
  }

  auto vec_0_entries = vectorEntries(vecs[0]);
  ASSERT_EQ(vec_0_entries.size(), 1);
  ASSERT_EQ(vec_0_entries.at(0), 1.0);
  
  auto vec_1_entries = vectorEntries(vecs[1]);
  ASSERT_EQ(vec_1_entries.size(), 2);
  ASSERT_EQ(vec_1_entries.at(0), 1.0);
  ASSERT_EQ(vec_1_entries.at(1), 1.0);
  
  auto vec_2_entries = vectorEntries(vecs[2]);
  ASSERT_EQ(vec_2_entries.size(), 3);
  ASSERT_EQ(vec_2_entries.at(0), 1.0);
  ASSERT_EQ(vec_2_entries.at(1), 1.0);
  ASSERT_EQ(vec_2_entries.at(2), 1.0);

  auto vec_3_entries = vectorEntries(vecs[3]);
  ASSERT_EQ(vec_3_entries.size(), 1);
  ASSERT_EQ(vec_3_entries.at(3), 1.0);
  
  auto vec_4_entries = vectorEntries(vecs[4]);
  ASSERT_EQ(vec_4_entries.size(), 1);
  ASSERT_EQ(vec_4_entries.at(4), 1.0);
}

TEST_F(CategoricalTrackingBlockTests, CorrectOutputWithGraph) {
  StringMatrix mock_data {
    {"id_1", "2022-01-01", "0"},
    {"id_1", "2022-01-01", "1"},
    {"id_2", "2022-01-02", "2"},
    {"id_3", "2022-01-07", "3"},
    {"id_2", "2022-01-07", "4"},
    {"id_1", "2022-01-07", "5"},
    {"id_4", "2022-01-07", "5"},
  };

  Graph graph {
    {"id_1", {"id_2", "id_3"}},
    {"id_2", {"id_1", "id_3"}},
    {"id_3", {"id_1", "id_2"}},
  };
  auto graph_ptr = std::make_shared<Graph>(std::move(graph));

  std::vector<SegmentedSparseFeatureVector> vecs;
  auto id_map = std::make_shared<StringToUidMap>(3);
  auto index = std::make_shared<CategoricalHistoryIndex>(/* n_ids = */ 3, /* n_categories = */ 6, /* buffer_size = */ 5);
  CategoricalTrackingBlock block(/* id_col = */ 0, /* timestamp_col = */ 1, /* category_col = */ 2, /* horizon = */ 0, /* lookback = */ 2, id_map, index, graph_ptr, /* max_n_neighbors = */ 1);

  for (auto& row : mock_data) {
    SegmentedSparseFeatureVector vec;
    addVectorSegmentWithBlock(block, row, vec);
    for (const auto [idx, _] : vectorEntries(vec)) {
      ASSERT_LT(idx, 12);
    }
    vecs.push_back(std::move(vec));
  }

  auto vec_0_entries = vectorEntries(vecs[0]);
  ASSERT_EQ(vec_0_entries.size(), 1);
  ASSERT_EQ(vec_0_entries.at(0), 1.0);
  
  auto vec_1_entries = vectorEntries(vecs[1]);
  ASSERT_EQ(vec_1_entries.size(), 2);
  ASSERT_EQ(vec_1_entries.at(0), 1.0);
  ASSERT_EQ(vec_1_entries.at(1), 1.0);
 
  auto vec_2_entries = vectorEntries(vecs[2]);
  ASSERT_EQ(vec_2_entries.size(), 3);
  ASSERT_EQ(vec_2_entries.at(2), 1.0);
  ASSERT_EQ(vec_2_entries.at(7), 1.0);
  ASSERT_EQ(vec_2_entries.at(8), 1.0);

  auto vec_3_entries = vectorEntries(vecs[3]);
  ASSERT_EQ(vec_3_entries.size(), 1);
  ASSERT_EQ(vec_3_entries.at(3), 1.0);
  
  auto vec_4_entries = vectorEntries(vecs[4]);
  // We don't expect an entry in position 9 for category with id "3"
  // tracked with "id_3" because max_n_neighbors = 1.
  ASSERT_EQ(vec_4_entries.size(), 1);
  ASSERT_EQ(vec_4_entries.at(4), 1.0);
  
  auto vec_5_entries = vectorEntries(vecs[5]);
  ASSERT_EQ(vec_5_entries.size(), 2);
  ASSERT_EQ(vec_5_entries.at(5), 1.0);
  ASSERT_EQ(vec_5_entries.at(11), 1.0);

  // We have already seen 3 classes so this new class is considered
  // "out-of-vocab" so the resulting vector segment will have no entry.
  auto vec_6_entries = vectorEntries(vecs[6]);
  ASSERT_EQ(vec_6_entries.size(), 0);
}



} // namespace thirdai::dataset