#include <gtest/gtest.h>
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/MachMemory.h>

namespace thirdai::data::tests {

TEST(MachMemorySerializationTest, BasicSerialization) {
  MachMemory original_samples("indices", "values", "labels", "doc_ids",
                              /*max_ids=*/10, /*max_samples_per_id=*/100);

  data::ColumnMap columns(
      {{"indices", data::ArrayColumn<uint32_t>::make(
                       {{1, 11, 21}, {2, 12, 22, 32, 42}, {3, 13, 23, 33}},
                       /*dim=*/100)},
       {"values", data::ArrayColumn<float>::make({{1.0, 2.0, 3.0},
                                                  {4.0, 5.0, 6.0, 7.0, 8.0},
                                                  {9.0, 10.0, 11.0, 12.0}})},
       {"labels", data::ArrayColumn<uint32_t>::make({{7, 8}, {3}, {4, 2, 9}},
                                                    /*dim=*/20)},
       {"doc_ids",
        data::ValueColumn<uint32_t>::make({100, 200, 100}, std::nullopt)}});

  original_samples.addSamples(columns);

  auto original_subset = original_samples.getSamples(10);

  auto new_samples = MachMemory::fromArchive(*original_samples.toArchive());

  ASSERT_EQ(original_samples.idToSamples(), new_samples->idToSamples());

  auto new_subset = new_samples->getSamples(10);

  ASSERT_EQ(new_subset->numRows(), original_subset->numRows());
}

}  // namespace thirdai::data::tests