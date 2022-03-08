#include <hashtable/src/TinyTable.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using thirdai::hashtable::TinyTable;

/**
 * Basic test that hashes elements with a simple hash function (i + j) % range,
 * inserts the elements into the TinyTable, and then ensures that the counts of
 * all hashes are as expected on retrieval from the TinyTable.
 */
TEST(TinyTableTest, InsertionAndQuery) {
  uint32_t num_tables = 100, range = 1000, num_elements = 100;

  // Element i table j is = 997 * (i + j) % range
  std::vector<std::vector<uint32_t>> expected_counts(
      range, std::vector<uint32_t>(range, 0));
  std::vector<uint32_t> hashes(num_tables * num_elements);
  for (uint32_t elem = 0; elem < num_elements; elem++) {
    for (uint32_t table = 0; table < num_tables; table++) {
      uint32_t hash = (997 * (elem + table)) % range;
      expected_counts.at(hash).at(elem)++;
      hashes.at(elem * num_tables + table) = hash;
    }
  }

  TinyTable<uint8_t> table(num_tables, range, num_elements, hashes);
  for (uint32_t hash = 0; hash < range; hash++) {
    std::vector<uint32_t> query(num_tables, hash);
    std::vector<uint32_t> query_counts(num_elements, 0);
    table.queryByCount(query, 0, query_counts);

    for (uint32_t i = 0; i < num_elements; i++) {
      ASSERT_EQ(query_counts.at(i), expected_counts.at(hash).at(i));
    }
  }
}