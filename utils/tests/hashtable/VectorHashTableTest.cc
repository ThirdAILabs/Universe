#include "../../hashtable/HashTable.h"
#include "../../hashtable/VectorHashTable.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cassert>

using thirdai::utils::VectorHashTable;

// TODO(josh): Abstract some of the logic from this test out and make the test
// smaller
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
  VectorHashTable<uint32_t> test_table(num_tables, table_range);

  // Generate hashes
  uint32_t hashes[num_tables * num_items];
  uint32_t labels[num_items];
  for (uint32_t i = 0; i < num_items; i++) {
    labels[i] = i + start_label;
    for (uint32_t t = 0; t < num_tables; t++) {
      uint32_t hash = (i + t) % table_range;
      hashes[t * num_items + i] = hash;
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
    test_table.queryByCount(&test_hashes[0], counts);

    // Vector query
    std::vector<uint32_t> result_vector(0);
    test_table.queryByVector(&test_hashes[0], result_vector);

    // Set query
    std::unordered_set<uint32_t> result_set(0);
    test_table.queryBySet(&test_hashes[0], result_set);

    for (uint32_t item = 0; item < num_items; item++) {
      uint32_t expected_count = 0;
      bool present = false;
      for (uint32_t table = 0; table < num_tables; table++) {
        if (hashes[table * num_items + item] == test_hash) {
          expected_count += 2;  // Because all items were inserted twice
          present = true;
        }
      }
      assert(counts[item] == expected_count);
      assert(std::count(result_vector.begin(), result_vector.end(), item) ==
             expected_count);
      assert(result_set.count(item) == present);
      result_set.erase(item);
    }

    assert(result_set.empty());
  }
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
  VectorHashTable<uint8_t> test_table(num_tables, table_range);

  // Add 5 items. Using a single table ensures they are inserted in the order
  // we intend even with a parallel insert.
  uint32_t hashes[] = {1, 1, 1, 2, 2};
  uint8_t labels[] = {5, 4, 3, 2, 1};
  test_table.insert(5, labels, hashes);

  // Sort buckets
  test_table.sortBuckets();

  // Do query in bucket 1
  std::vector<uint8_t> result(0);
  uint32_t test_hashes_1[] = {1};
  test_table.queryByVector(test_hashes_1, result);
  assert(result.size() == 3);
  assert(result[0] == 3 && result[1] == 4 && result[2] == 5);

  // Do query in bucket 2
  result.clear();
  uint32_t test_hashes_2[] = {2};
  test_table.queryByVector(test_hashes_2, result);
  assert(result.size() == 2);
  assert(result[0] == 1 && result[1] == 2);

  // Clear table
  test_table.clearTables();

  // Query both bucket 1 and 2
  result.clear();
  test_table.queryByVector(test_hashes_1, result);
  test_table.queryByVector(test_hashes_2, result);
  assert(result.empty());
}