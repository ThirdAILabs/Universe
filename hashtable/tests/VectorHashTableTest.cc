#include <hashtable/src/HashTable.h>
#include <hashtable/src/VectorHashTable.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

using thirdai::hashtable::VectorHashTable;

/**
 * This test creates a VectorHashTable with N tables and a hash range of R,
 * then inserts m elements, where element i has hashes i % R, (i + 1) % R,
 * ..., (i + N - 1) % R, individually, inserts all of the elements sequentially
 * and as a normal batch, then makes sure they can be succesfully retrieved
 * the correct number of times with all query methods.
 */
TEST(VectorHashTableTest, ExactRetrievalTest) {
  uint32_t num_tables = 5;
  uint32_t table_range = 100;
  uint32_t start_label = 0;
  uint32_t num_items = 100;
  VectorHashTable test_table(num_tables, table_range);

  // Generate hashes
  uint32_t* hashes = new uint32_t[num_tables * num_items];
  uint32_t* labels = new uint32_t[num_items];
  for (uint32_t i = 0; i < num_items; i++) {
    labels[i] = i + start_label;
    for (uint32_t t = 0; t < num_tables; t++) {
      uint32_t hash = (i + t) % table_range;
      hashes[num_tables * i + t] = hash;
    }
  }

  // Insert twice
  test_table.insertSequential(num_items, start_label, hashes);
  test_table.insert(num_items, labels, hashes);

  // Check some basic queries
  for (uint32_t test_hash = 0; test_hash < table_range; test_hash++) {
    std::vector<uint32_t> test_hashes(num_tables, test_hash);

    // Count query
    std::vector<uint32_t> counts(num_items, 0);
    test_table.queryByCount(test_hashes.data(), counts);

    // Vector query
    std::vector<uint32_t> result_vector(0);
    test_table.queryByVector(test_hashes.data(), result_vector);

    // Set query
    std::unordered_set<uint32_t> result_set(0);
    test_table.queryBySet(test_hashes.data(), result_set);

    for (uint32_t item = 0; item < num_items; item++) {
      uint32_t expected_count = 0;
      bool present = false;
      for (uint32_t table = 0; table < num_tables; table++) {
        if (hashes[num_tables * item + table] == test_hash) {
          expected_count += 2;  // Because all items were inserted twice
          present = true;
        }
      }
      ASSERT_EQ(counts[item], expected_count);
      ASSERT_EQ(std::count(result_vector.begin(), result_vector.end(), item),
                expected_count);
      ASSERT_EQ(result_set.count(item), present);
      result_set.erase(item);
    }

    ASSERT_TRUE(result_set.empty());
  }
  delete[] hashes;
  delete[] labels;
}

/**
 * Tests whether buckets are correctly sorted by inserting 5 elements the
 * table, calling sort, and ensuring that the vector of returned elements
 * is as expected.
 */
TEST(VectorHashTableTest, SortAndClearBucketsTest) {
  // Create a hash table with a single table
  uint32_t num_tables = 1;
  uint32_t table_range = 10;
  VectorHashTable test_table(num_tables, table_range);

  // Add 5 items. Using a single table ensures they are inserted in the order
  // we intend even with a parallel insert.
  uint32_t hashes[] = {1, 1, 1, 2, 2};
  uint32_t labels[] = {5, 4, 3, 2, 1};
  test_table.insert(5, labels, hashes);

  // Sort buckets
  test_table.sortBuckets();

  // Do query in bucket 1
  std::vector<uint32_t> result(0);
  uint32_t test_hashes_1[] = {1};
  test_table.queryByVector(test_hashes_1, result);
  ASSERT_EQ(result.size(), 3);
  ASSERT_EQ(result[0], 3);
  ASSERT_EQ(result[1], 4);
  ASSERT_EQ(result[2], 5);

  // Do query in bucket 2
  result.clear();
  uint32_t test_hashes_2[] = {2};
  test_table.queryByVector(test_hashes_2, result);
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 1);
  ASSERT_EQ(result[1], 2);

  // Clear table
  test_table.clearTables();

  // Query both bucket 1 and 2
  result.clear();
  test_table.queryByVector(test_hashes_1, result);
  test_table.queryByVector(test_hashes_2, result);
  ASSERT_TRUE(result.empty());
}

/**
 * Tests reservoir sampling by adding 100 items to a max reservoir size 10
 * table and ensuring we get back exactly 10 items, some of which are not the
 * first 10 we added (the chance we would randomly select the first 10 out of
 * 100 is vanishingly small).
 */
TEST(VectorHashTableTest, SimpleReservoirTest) {
  // Create a hash table with a single table
  uint32_t num_tables = 1;
  uint32_t table_range = 100;
  uint32_t max_reservoir_size = 10;
  uint32_t num_elements_to_add = 100;
  uint32_t element_hash = 42;
  uint32_t seed = 43;
  VectorHashTable test_table(num_tables, table_range, max_reservoir_size, seed);

  // Add 100 items all with the same hash. Using a single table ensures they are
  // inserted in the order we intend even with a parallel insert.
  auto hashes = std::vector<uint32_t>(num_elements_to_add, element_hash);
  auto labels = std::vector<uint32_t>(num_elements_to_add);
  std::iota(std::begin(labels), std::end(labels), 0);
  test_table.insert(num_elements_to_add, labels.data(), hashes.data());

  // Do query
  std::vector<uint32_t> result(0);
  uint32_t test_hashes_1[] = {element_hash};
  test_table.queryByVector(test_hashes_1, result);
  ASSERT_EQ(result.size(), 10);
  std::sort(result.begin(), result.end());
  ASSERT_TRUE(!(result[0] == 0 && result[9] == 9));
}