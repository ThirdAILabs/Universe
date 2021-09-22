#include "../../hashtable/SampledHashTable.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

TEST(SampledHashTableTest, InsertionQueryWithoutReservoirSampling) {
  uint32_t num_tables = 100, range_pow = 10, reservoir_size = 100,
           num_inserts = 10000;
  uint32_t range = 1 << range_pow;
  thirdai::utils::SampledHashTable<uint32_t> table(num_tables, reservoir_size,
                                                   range_pow);

  std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>> insertions(
      num_tables);
  std::default_random_engine rd;
  std::uniform_int_distribution<uint32_t> dist(0, range - 1);

  uint32_t* labels = new uint32_t[num_inserts];
  uint32_t* hashes = new uint32_t[num_tables * num_inserts];

  for (uint32_t i = 0; i < num_inserts; i++) {
    labels[i] = i;
    for (uint32_t t = 0; t < num_tables; t++) {
      uint32_t hash = dist(rd);
      while (insertions.at(t)[hash].size() >= reservoir_size) {
        hash = (hash + 1) % range;
      }
      insertions.at(t)[hash].push_back(i);

      hashes[i * num_tables + t] = hash;
    }
  }
  table.insert(num_inserts, labels, hashes);

  for (uint32_t i = 0; i < num_inserts; i++) {
    for (uint32_t t = 0; t < num_tables; t++) {
      uint32_t hash = dist(rd);
      while (insertions.at(t)[hash].size() >= reservoir_size) {
        hash = (hash + 1) % range;
      }
      insertions.at(t)[hash].push_back(i + num_inserts);

      hashes[i * num_tables + t] = hash;
    }
  }
  table.insertSequential(num_inserts, num_inserts, hashes);

  delete[] labels;
  delete[] hashes;

  uint32_t num_queries = 1;

  for (uint32_t q = 0; q < num_queries; q++) {
    uint32_t query_hashes[num_tables];
    std::vector<uint32_t> exp_counts(2 * num_inserts, 0);
    std::vector<uint32_t> exp_vec_results;
    std::unordered_set<uint32_t> exp_set_results;
    for (uint32_t t = 0; t < num_tables; t++) {
      uint32_t hash = dist(rd);
      for (uint32_t x : insertions.at(t)[hash]) {
        exp_vec_results.push_back(x);
        exp_set_results.insert(x);
        exp_counts[x]++;
      }
      query_hashes[t] = hash;
    }

    std::vector<uint32_t> counts(2 * num_inserts, 0);
    table.queryByCount(query_hashes, counts);
    for (uint32_t i = 0; i < 2 * num_inserts; i++) {
      ASSERT_EQ(counts[i], exp_counts[i]);
    }

    std::vector<uint32_t> vec_results;
    table.queryByVector(query_hashes, vec_results);
    std::sort(vec_results.begin(), vec_results.end());
    std::sort(exp_vec_results.begin(), exp_vec_results.end());

    ASSERT_EQ(vec_results.size(), exp_vec_results.size());
    for (uint32_t i = 0; i < vec_results.size(); i++) {
      ASSERT_EQ(vec_results[i], exp_vec_results[i]);
    }

    std::unordered_set<uint32_t> set_results;
    table.queryBySet(query_hashes, set_results);
    ASSERT_EQ(set_results.size(), exp_set_results.size());
    for (uint32_t x : set_results) {
      ASSERT_TRUE(exp_set_results.count(x));
    }
    for (uint32_t x : exp_set_results) {
      ASSERT_TRUE(set_results.count(x));
    }
  }
}

TEST(SampledHashTableTest, InsertionQueryWithReservoirSampling) {
  uint32_t num_tables = 100, range_pow = 10, reservoir_size = 10,
           num_inserts = 10000;
  uint32_t range = 1 << range_pow;
  thirdai::utils::SampledHashTable<uint32_t> table(num_tables, reservoir_size,
                                                   range_pow);

  std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>> insertions(
      num_tables);
  std::default_random_engine rd;
  std::uniform_int_distribution<uint32_t> dist(0, range - 1);

  uint32_t* labels = new uint32_t[num_inserts];
  uint32_t* hashes = new uint32_t[num_tables * num_inserts];

  for (uint32_t i = 0; i < num_inserts; i++) {
    labels[i] = i;
    for (uint32_t t = 0; t < num_tables; t++) {
      uint32_t hash = dist(rd);
      insertions.at(t)[hash].push_back(i);

      hashes[i * num_tables + t] = hash;
    }
  }
  table.insert(num_inserts, labels, hashes);

  for (uint32_t i = 0; i < num_inserts; i++) {
    for (uint32_t t = 0; t < num_tables; t++) {
      uint32_t hash = dist(rd);
      insertions.at(t)[hash].push_back(i + num_inserts);

      hashes[i * num_tables + t] = hash;
    }
  }
  table.insertSequential(num_inserts, num_inserts, hashes);

  delete[] labels;
  delete[] hashes;

  uint32_t num_queries = 1000;

  for (uint32_t q = 0; q < num_queries; q++) {
    uint32_t query_hashes[num_tables];
    std::vector<uint32_t> exp_counts(2 * num_inserts, 0);
    std::unordered_set<uint32_t> exp_set_results;
    for (uint32_t t = 0; t < num_tables; t++) {
      uint32_t hash = dist(rd);
      for (uint32_t x : insertions.at(t)[hash]) {
        exp_set_results.insert(x);
        exp_counts[x]++;
      }
      query_hashes[t] = hash;
    }

    std::vector<uint32_t> counts(2 * num_inserts, 0);
    table.queryByCount(query_hashes, counts);
    for (uint32_t i = 0; i < 2 * num_inserts; i++) {
      ASSERT_LE(counts[i], exp_counts[i]);
    }

    std::vector<uint32_t> vec_results;
    table.queryByVector(query_hashes, vec_results);
    for (uint32_t vec_result : vec_results) {
      ASSERT_TRUE(exp_set_results.count(vec_result));
    }

    std::unordered_set<uint32_t> set_results;
    table.queryBySet(query_hashes, set_results);
    for (uint32_t x : set_results) {
      ASSERT_TRUE(exp_set_results.count(x));
    }
  }
}