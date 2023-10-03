#include "gtest/gtest.h"
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/StringCast.h>
#include <data/tests/MockDataSource.h>
#include <optional>
#include <sstream>
#include <unordered_set>

namespace thirdai::data::tests {

uint32_t tokenValue(size_t row, size_t col) { return row + col; }

float decimalValue(size_t row, size_t col) {
  return static_cast<float>(row + col + 1) / 4;
}

DataSourcePtr getMockDataSource(size_t n_lines) {
  std::vector<std::string> lines = {"token,tokens,decimal,decimals"};

  for (size_t i = 0; i < n_lines; i++) {
    // Token
    std::stringstream line;
    line << tokenValue(i, 0) << ",";

    // Tokens
    for (size_t j = 0; j < (i % 10); j++) {
      if (j > 0) {
        line << " ";
      }
      line << tokenValue(i, j + 1);
    }

    // Decimal
    line << "," << decimalValue(i, 0) << ",";

    // Decimals
    for (size_t j = 0; j < (i % 10); j++) {
      if (j > 0) {
        line << " ";
      }
      line << decimalValue(i, j + 1);
    }

    lines.push_back(line.str());
  }

  return std::make_shared<MockDataSource>(std::move(lines));
}

TEST(DataLoaderTest, Streaming) {
  size_t n_full_chunks = 4, n_batches = 10, batch_size = 20;
  size_t partial_chunk_full_batches = 4, last_batch_size = 7;
  size_t n_rows = n_full_chunks * n_batches * batch_size +
                  partial_chunk_full_batches * batch_size + last_batch_size;

  auto data_iterator = CsvIterator::make(getMockDataSource(n_rows),
                                         /* delimiter= */ ',',
                                         /* rows_per_load= */ 64);

  auto transformations =
      Pipeline::make()
          ->then(std::make_shared<StringToToken>("token", "token_cast", n_rows))
          ->then(std::make_shared<StringToTokenArray>("tokens", "tokens_cast",
                                                      ' ', n_rows + 10))
          ->then(std::make_shared<StringToDecimal>("decimal", "decimal_cast"))
          ->then(std::make_shared<StringToDecimalArray>(
              "decimals", "decimals_cast", ' ', std::nullopt));

  auto loader = Loader::make(
      data_iterator, transformations, std::make_shared<State>(),
      {{"tokens_cast", "decimals_cast"}}, {{"token_cast", "decimal_cast"}},
      /* batch_size= */ batch_size, /* shuffle= */ true, /* verbose= */ true,
      /* shuffle_buffer_size= */ 50);

  std::unordered_set<uint32_t> rows_seen;

  for (size_t c = 0; c < n_full_chunks + 1; c++) {
    auto chunk = loader->next(n_batches);
    ASSERT_TRUE(chunk.has_value());

    auto [data, labels] = std::move(*chunk);

    size_t expected_batches =
        (c < n_full_chunks) ? n_batches : partial_chunk_full_batches + 1;
    ASSERT_EQ(data.size(), expected_batches);
    ASSERT_EQ(labels.size(), expected_batches);

    for (size_t b = 0; b < data.size(); b++) {
      ASSERT_EQ(data[b].size(), 1);
      ASSERT_EQ(labels[b].size(), 1);

      size_t expected_batches =
          (c < n_full_chunks || b < partial_chunk_full_batches)
              ? batch_size
              : last_batch_size;

      ASSERT_EQ(data[b][0]->batchSize(), expected_batches);
      ASSERT_EQ(labels[b][0]->batchSize(), expected_batches);

      for (size_t i = 0; i < data[b][0]->batchSize(); i++) {
        const BoltVector& data_vec = data[b][0]->getVector(i);
        ASSERT_FALSE(data_vec.isDense());
        ASSERT_FALSE(data_vec.hasGradients());

        const BoltVector& label_vec = labels[b][0]->getVector(i);
        ASSERT_FALSE(label_vec.isDense());
        ASSERT_FALSE(label_vec.hasGradients());
        ASSERT_EQ(label_vec.len, 1);

        uint32_t row_id = label_vec.active_neurons[0];

        ASSERT_EQ(label_vec.activations[0], decimalValue(row_id, 0));

        ASSERT_EQ(data_vec.len, row_id % 10);

        for (size_t j = 0; j < data_vec.len; j++) {
          ASSERT_EQ(data_vec.active_neurons[j], tokenValue(row_id, j + 1));
          ASSERT_EQ(data_vec.activations[j], decimalValue(row_id, j + 1));
        }

        rows_seen.insert(row_id);
      }
    }
  }

  ASSERT_EQ(rows_seen.size(), n_rows);

  ASSERT_FALSE(loader->next(10).has_value());
}

}  // namespace thirdai::data::tests