#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/tests/MockDataSource.h>
#include <memory>
#include <string>

namespace thirdai::data::tests {

void checkStringColumnContents(const ColumnPtr& column,
                               const std::vector<std::string>& contents) {
  auto str_col = std::dynamic_pointer_cast<ValueColumn<std::string>>(column);
  ASSERT_TRUE(str_col);

  ASSERT_EQ(str_col->data(), contents);
}

void checkReturnedRows(ColumnMapIterator& iterator) {
  auto columns_1 = iterator.next();
  ASSERT_TRUE(columns_1.has_value());
  ASSERT_EQ(columns_1->columns().size(), 3);
  checkStringColumnContents(columns_1->getColumn("col_a"), {"1", "4"});
  checkStringColumnContents(columns_1->getColumn("col_b"), {"2", "5"});
  checkStringColumnContents(columns_1->getColumn("col_c"), {"3", "6"});

  auto columns_2 = iterator.next();
  ASSERT_TRUE(columns_2.has_value());
  ASSERT_EQ(columns_2->columns().size(), 3);
  checkStringColumnContents(columns_2->getColumn("col_a"), {"7"});
  checkStringColumnContents(columns_2->getColumn("col_b"), {"8"});
  checkStringColumnContents(columns_2->getColumn("col_c"), {"9"});

  auto columns_3 = iterator.next();
  ASSERT_FALSE(columns_3.has_value());
}

TEST(ColumnMapIteratorTests, CsvIterator) {
  auto data_source = std::make_shared<MockDataSource>(
      std::vector<std::string>{"col_a,col_b,col_c", "1,2,3", "4,5,6", "7,8,9"});

  CsvIterator iterator(data_source, ',', /* rows_per_load= */ 2);

  checkReturnedRows(iterator);
}

TEST(ColumnMapIteratorTests, JsonIterator) {
  auto data_source = std::make_shared<MockDataSource>(
      std::vector<std::string>{R"({"col_a":"1", "col_b":"2", "col_c":"3"})",
                               R"({"col_a":"4", "col_b":"5", "col_c":"6"})",
                               R"({"col_a":"7", "col_b":"8", "col_c":"9"})"});

  JsonIterator iterator(data_source, {"col_a", "col_b", "col_c"},
                        /* rows_per_load= */ 2);

  checkReturnedRows(iterator);
}

}  // namespace thirdai::data::tests