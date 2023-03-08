#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/Shuffler.h>
#include <cstddef>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace thirdai::dataset {

size_t NUM_ELEMENTS = 1000;
size_t BATCH_SIZE = 128;
size_t NUM_COLUMNS = 5;
size_t NUM_BATCHES = (NUM_ELEMENTS + BATCH_SIZE - 1) / BATCH_SIZE;

size_t batchSize(size_t batch_id) {
  size_t num_full_batches = NUM_ELEMENTS / BATCH_SIZE;
  if (batch_id < num_full_batches) {
    return BATCH_SIZE;
  }
  return NUM_ELEMENTS % BATCH_SIZE;
}

BatchColumns batchColumns() {
  BatchColumns columns(NUM_COLUMNS, std::vector<BoltBatch>(NUM_BATCHES));
  for (size_t batch_id = 0; batch_id < NUM_BATCHES; batch_id++) {
    for (size_t column_id = 0; column_id < NUM_COLUMNS; column_id++) {
      columns[column_id][batch_id] = std::move(BoltBatch(batchSize(batch_id)));
    }
  }

  for (uint32_t id = 0; id < NUM_ELEMENTS; id++) {
    size_t batch_id = id / BATCH_SIZE;
    size_t vec_id = id % BATCH_SIZE;
    for (uint32_t column_id = 0; column_id < NUM_COLUMNS; column_id++) {
      columns[column_id][batch_id][vec_id] = BoltVector::makeSparseVector(
          /* indices= */ {id, column_id}, /* values= */ {1.0, 1.0});
    }
  }

  return columns;
}

auto shuffledColumns(uint32_t seed = 100) {
  auto columns = batchColumns();
  std::mt19937 gen(seed);
  shuffleInPlace(columns, gen);
  return columns;
}

TEST(ShufflerTests, HasAllOriginalElements) {
  auto columns = shuffledColumns();

  std::unordered_set<uint32_t> elements;

  for (auto& column : columns) {
    for (auto& batch : column) {
      for (BoltVector& vec : batch) {
        uint32_t sample_id = vec.active_neurons[0];
        uint32_t column_id = vec.active_neurons[1];
        elements.insert(column_id * NUM_ELEMENTS + sample_id);
      }
    }
  }

  ASSERT_EQ(elements.size(), NUM_ELEMENTS * NUM_COLUMNS);
}

TEST(ShufflerTests, CorrectBatchSizes) {
  auto columns = shuffledColumns();

  for (auto& column : columns) {
    for (size_t batch_id = 0; batch_id < column.size(); batch_id++) {
      ASSERT_EQ(column[batch_id].getBatchSize(), batchSize(batch_id));
    }
  }
}

TEST(ShufflerTests, ConsistentOrderingAcrossColumns) {
  auto columns = shuffledColumns();

  for (size_t batch_id = 0; batch_id < NUM_BATCHES; batch_id++) {
    for (size_t vec_id = 0; vec_id < batchSize(batch_id); vec_id++) {
      size_t element_id = columns.front()[batch_id][vec_id].active_neurons[0];
      for (auto& column : columns) {
        ASSERT_EQ(column[batch_id][vec_id].active_neurons[0], element_id);
      }
    }
  }
}

TEST(ShufflerTests, VectorStaysInSameColumn) {
  auto columns = shuffledColumns();

  for (size_t column_id = 0; column_id < NUM_COLUMNS; column_id++) {
    for (const auto& batch : columns[column_id]) {
      for (const auto& vec : batch) {
        ASSERT_EQ(vec.active_neurons[1], column_id);
      }
    }
  }

  for (size_t batch_id = 0; batch_id < NUM_BATCHES; batch_id++) {
    for (size_t vec_id = 0; vec_id < batchSize(batch_id); vec_id++) {
      size_t element_id = columns.front()[batch_id][vec_id].active_neurons[0];
      for (auto& column : columns) {
        ASSERT_EQ(column[batch_id][vec_id].active_neurons[0], element_id);
      }
    }
  }
}

uint32_t absoluteDifference(uint32_t lhs, uint32_t rhs) {
  if (lhs > rhs) {
    return lhs - rhs;
  }
  return rhs - lhs;
}

TEST(ShufflerTests, ShuffledEnough) {
  auto columns = shuffledColumns();
  std::mt19937 gen(100);

  double total_distance = 0;

  for (uint32_t batch_id = 0; batch_id < NUM_BATCHES; batch_id++) {
    auto vec_id_permutation = permutation(batchSize(batch_id), gen);
    for (uint32_t vec_id = 0; vec_id < batchSize(batch_id); vec_id++) {
      uint32_t id = batch_id * BATCH_SIZE + vec_id;

      // We randomly permute vectors within the same batch to simulate the
      // shuffling that happens when bolt processes a batch in parallel.
      uint32_t final_vec_id = vec_id_permutation[vec_id];
      // uint32_t final_vec_id = vec_id;
      uint32_t id_in_vector =
          columns.front()[batch_id][final_vec_id].active_neurons[0];
      std::cout << id_in_vector << std::endl;
      total_distance += absoluteDifference(id, id_in_vector);
    }
    std::cout << "==========" << std::endl;
  }

  double average_distance = total_distance / NUM_ELEMENTS;
  std::cout << "AVERAGE DISTANCE: " << average_distance << std::endl;
  ASSERT_GE(average_distance, NUM_ELEMENTS / 3);
}

bool sameOrdering(const BatchColumns& lhs, const BatchColumns& rhs) {
  for (size_t batch_id = 0; batch_id < NUM_BATCHES; batch_id++) {
    for (size_t vec_id = 0; vec_id < batchSize(batch_id); vec_id++) {
      if (lhs.front()[batch_id][vec_id].active_neurons[0] !=
          rhs.front()[batch_id][vec_id].active_neurons[0]) {
        return false;
      }
    }
  }
  return true;
}

TEST(ShufflerTests, Deterministic) {
  auto seeded_columns_1 = shuffledColumns(/* seed= */ 10);
  auto seeded_columns_2 = shuffledColumns(/* seed= */ 10);
  auto seeded_columns_3 = shuffledColumns(/* seed= */ 100);

  ASSERT_TRUE(sameOrdering(seeded_columns_1, seeded_columns_2));
  ASSERT_FALSE(sameOrdering(seeded_columns_1, seeded_columns_3));
}

}  // namespace thirdai::dataset