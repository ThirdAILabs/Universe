#include <gtest/gtest.h>
#include <auto_ml/src/rlhf/BalancingSamples.h>
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>

namespace thirdai::automl::tests {

TEST(BalancingSamplesSerializationTest, BasicSerialization) {
  udt::BalancingSamples original_balancing_samples(
      "indices", "values", "labels", "doc_ids",
      /*indices_dim=*/100, /*label_dim=*/20,
      /*max_docs=*/10, /*max_samples_per_doc=*/100);

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

  original_balancing_samples.addSamples(columns);

  auto original_samples = original_balancing_samples.balancingSamples(10);

  udt::BalancingSamples new_balancing_samples(
      *original_balancing_samples.toArchive());

  ASSERT_EQ(original_balancing_samples.samplesPerDoc(),
            new_balancing_samples.samplesPerDoc());

  auto new_samples = new_balancing_samples.balancingSamples(10);

  ASSERT_EQ(original_samples.numRows(), new_samples.numRows());
}

}  // namespace thirdai::automl::tests