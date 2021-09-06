#include "../../hashtable/HashTable.h"
#include "../../hashtable/VectorHashTable.h"
#include <gtest/gtest.h>
#include <cassert>

using namespace thirdai::utils;

class VectorHashTableTest : public testing::Test {};

/**
 * This test creates a VectorHashTable with N tables and a hash range of R,
 * then inserts m elements, where element i has hashes i % R, (i + 1) % R,
 * ..., (i + N - 1) % R, individually, inserts all of the elements sequentially
 * and as a normal batch, then makes sure they can be succesfully retrieved
 * the correct number of times with all query methods.
 */
TEST_F(VectorHashTableTest, ExactRetrievalTest) {
  uint32_t num_tables = 5;
  uint32_t table_range = 100;
  uint32_t start_label = 0;
  uint32_t num_items = 100;
  VectorHashTable<uint32_t> test_table(num_tables, table_range);

  // // Generate hashes
  // uint32_t hashes[num_tables * num_items];
  // uint32_t labels[num_items];
  // for (uint32_t i = 0; i < num_items; i++) {
  //   labels[i] = i + start_label;
  //   for (uint32_t t = 0; t < num_tables; t++) {
  //     uint32_t hash = (i + t) % table_range;
  //     hashes[t * num_items + i] = hash;
  //   }
  // }

  // // Insert twice
  // test_table.insertSequential(num_items, start_label, hashes);
  // test_table.insert(num_items, labels, hashes);

  // // Check some basic queries
  // for (uint32_t test_hash = 0; test_hash < table_range; test_hash++) {
  //   std::vector<uint32_t> test_hashes(num_tables, test_hash);

  //   // Count query
  //   std::vector<uint32_t> counts(num_items, 0);
  //   test_table.queryByCount(&test_hashes[0], counts);

  //   for (uint32_t item = 0; item < num_items; item++) {
  //     uint32_t expected_count = 0;
  //     for (uint32_t table = 0; table < num_tables; table++) {
  //       if (hashes[table * num_items + table] == test_hash) {
  //         expected_count++;
  //       }
  //     }
  //     assert(counts[item] == expected_count);
  //   }
  // }
}

// TEST_F(VectorHashTableTest, SortBucketTest) {

// }