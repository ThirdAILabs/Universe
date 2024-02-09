#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Tabular.h>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::data::tests {

ColumnMap makeColumnMap(
    const std::unordered_map<std::string, std::vector<std::string>>& columns) {
  std::unordered_map<std::string, ColumnPtr> column_map;

  for (auto [name, data] : columns) {
    column_map[name] = ValueColumn<std::string>::make(std::move(data));
  }

  return ColumnMap(column_map);
}

const std::string OUTPUT = "__tabular_columns__";

std::vector<std::vector<uint32_t>> getTabularTokens(const ColumnMap& columns) {
  return std::dynamic_pointer_cast<ArrayColumn<uint32_t>>(
             columns.getColumn(OUTPUT))
      ->data();
}

void testSameValueDifferentColumns(
    const std::unordered_map<std::string, std::vector<std::string>>& columns,
    const std::vector<NumericalColumn>& numerical_columns,
    const std::vector<CategoricalColumn>& categorical_columns) {
  Tabular transform(numerical_columns, categorical_columns, OUTPUT,
                    /* cross_column_pairgrams= */ false);

  auto output =
      getTabularTokens(transform.applyStateless(makeColumnMap(columns)));

  for (const auto& row : output) {
    std::unordered_set<uint32_t> unique_items(row.begin(), row.end());
    ASSERT_EQ(unique_items.size(), 2);
  }
}

TEST(TabularTests, SameCategoricalValueDifferentColumns) {
  testSameValueDifferentColumns(
      {{"a", {"aa", "bb", "cc"}}, {"b", {"aa", "bb", "cc"}}}, {},
      {CategoricalColumn("a"), CategoricalColumn("b")});
}

TEST(TabularTests, SameNumericalValueDifferentColumns) {
  testSameValueDifferentColumns(
      {{"a", {"0.5", "1.5", "2.5"}}, {"b", {"0.5", "1.5", "2.5"}}},
      {NumericalColumn("a", 0, 3, 3), NumericalColumn("b", 0, 3, 3)}, {});
}

void testValuesInSingleColumn(
    const std::unordered_map<std::string, std::vector<std::string>>& columns,
    const std::vector<NumericalColumn>& numerical_columns,
    const std::vector<CategoricalColumn>& categorical_columns) {
  Tabular transform(numerical_columns, categorical_columns, OUTPUT,
                    /* cross_column_pairgrams= */ false);

  auto output =
      getTabularTokens(transform.applyStateless(makeColumnMap(columns)));

  for (const auto& row : output) {
    ASSERT_EQ(row.size(), 1);
  }

  ASSERT_EQ(output.size() % 2, 0);

  std::unordered_set<uint32_t> unique_items;
  for (size_t i = 0; i < output.size() / 2; i++) {
    ASSERT_EQ(output.at(i), output.at(output.size() / 2 + i));
    unique_items.insert(output.at(i).at(0));
  }

  ASSERT_EQ(unique_items.size(), output.size() / 2);
}

TEST(TabularTests, CategoricalValueSingleColumn) {
  testValuesInSingleColumn({{"a", {"aa", "bb", "cc", "aa", "bb", "cc"}}}, {},
                           {CategoricalColumn("a")});
}

TEST(TabularTests, NumericalValueSingleColumn) {
  testValuesInSingleColumn({{"a", {"0.5", "1.5", "2.5", "0.2", "1.7", "2.8"}}},
                           {NumericalColumn("a", 0, 3, 3)}, {});
}

size_t intersectionSize(const std::vector<uint32_t>& a,
                        const std::vector<uint32_t>& b) {
  std::unordered_set<uint32_t> a_set(a.begin(), a.end());

  size_t cnt = 0;
  for (uint32_t item : b) {
    if (a_set.count(item)) {
      cnt++;
    }
  }

  return cnt;
}

TEST(TabularTests, PairgramConsistency) {
  auto columns = makeColumnMap(
      {{"a", {"aa", "bb", "aa", "bb"}}, {"b", {"0.5", "1.5", "1.4", "1.9"}}});

  Tabular transform({NumericalColumn("b", 0, 2, 2)}, {CategoricalColumn("a")},
                    OUTPUT, /* cross_column_pairgrams= */ true);

  auto output = getTabularTokens(transform.applyStateless(columns));

  ASSERT_EQ(intersectionSize(output.at(0), output.at(1)), 0);
  ASSERT_EQ(intersectionSize(output.at(0), output.at(2)), 1);
  ASSERT_EQ(intersectionSize(output.at(1), output.at(2)), 1);
  ASSERT_EQ(output.at(1), output.at(3));
  ASSERT_EQ(intersectionSize(output.at(2), output.at(3)), 1);
}

TEST(TabularTests, HashDistribution) {
  size_t rows = 100, cols = 100;

  std::vector<CategoricalColumn> cat_cols;
  std::unordered_map<std::string, std::vector<std::string>> columns;

  for (size_t i = 0; i < cols; i++) {
    std::string name = "col_" + std::to_string(i);
    cat_cols.push_back(CategoricalColumn(name));
    for (size_t j = i; j < rows + i; j++) {
      columns[name].push_back(std::to_string(j));
    }
  }

  Tabular transform({}, cat_cols, OUTPUT, /* cross_column_pairgrams= */ true);

  auto output =
      getTabularTokens(transform.applyStateless(makeColumnMap(columns)));

  std::unordered_map<uint32_t, size_t> hash_counts;

  for (const auto& row : output) {
    for (const auto& item : row) {
      hash_counts[item]++;
    }
  }

  for (const auto& [_, cnt] : hash_counts) {
    ASSERT_LE(cnt, 4);  // No more than 4 collisions.
  }
}

TEST(TabularTests, Serialization) {
  auto columns =
      makeColumnMap({{"a", {"aa", "bb", "cc", "aa", "bb", "cc"}},
                     {"b", {"0.5", "1.5", "2.5", "0.2", "1.7", "2.8"}},
                     {"c", {"aa", "cc", "bb", "bb", "bb", "aa"}}});

  Tabular transform({NumericalColumn("b", 0, 3, 3)},
                    {CategoricalColumn("a"), CategoricalColumn("c")}, OUTPUT,
                    /* cross_column_pairgrams= */ true);

  // We down cast to transformation because otherwise it was trying to call
  // the cereal "serialize" method. This can be removed once cereal is
  // officially deprecated.
  auto transform_copy = Transformation::deserialize(
      dynamic_cast<Transformation*>(&transform)->serialize());

  auto output1 = getTabularTokens(transform.applyStateless(columns));
  auto output2 = getTabularTokens(transform_copy->applyStateless(columns));

  ASSERT_EQ(output1, output2);
}

}  // namespace thirdai::data::tests