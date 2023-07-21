#include "gtest/gtest.h"
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <numeric>

namespace thirdai::data::tests {

void runConversionTest(bool specify_values) {
  std::vector<std::vector<uint32_t>> indices;
  std::vector<std::vector<float>> values;

  std::vector<size_t> row_lens = {4, 7, 6, 4, 12, 15, 8, 3};

  size_t value = 0;
  for (size_t len : row_lens) {
    std::vector<uint32_t> row_indices(len);
    std::iota(row_indices.begin(), row_indices.end(), value);
    indices.push_back(row_indices);

    std::vector<float> row_values(len);
    std::iota(row_values.begin(), row_values.end(), static_cast<float>(value));
    values.push_back(row_values);

    value += len;
  }

  auto indices_copy = indices;
  auto values_copy = values;

  auto indices_col = ArrayColumn<uint32_t>::make(std::move(indices_copy), 1000);
  auto values_col = ArrayColumn<float>::make(std::move(values_copy));

  ColumnMap columns({{"indices", indices_col}, {"values", values_col}});

  std::optional<std::string> values_col_name =
      specify_values ? std::make_optional("values") : std::nullopt;
  auto tensors = convertToTensors(columns, {{"indices", values_col_name}}, 3);

  size_t row_cnt = 0;
  size_t value_cnt = 0;
  for (const auto& batch : tensors) {
    for (size_t i = 0; i < batch[0]->batchSize(); i++) {
      const BoltVector& vec = batch[0]->getVector(i);

      EXPECT_EQ(indices.at(row_cnt).size(), vec.len);

      for (size_t j = 0; j < vec.len; j++) {
        EXPECT_EQ(vec.active_neurons[j], value_cnt);
        if (specify_values) {
          EXPECT_EQ(vec.activations[j], static_cast<float>(value_cnt));
        } else {
          EXPECT_EQ(vec.activations[i], 1.0);
        }
        value_cnt++;
      }
      row_cnt++;
    }
  }
}

TEST(TensorConversionTests, WithValues) {
  runConversionTest(/* specify_values= */ true);
}

TEST(TensorConversionTests, WithoutValues) {
  runConversionTest(/* specify_values= */ false);
}

}  // namespace thirdai::data::tests