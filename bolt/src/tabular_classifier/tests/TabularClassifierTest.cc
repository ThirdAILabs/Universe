#include <bolt/src/tabular_classifier/TabularClassifier.h>
#include <gtest/gtest.h>

namespace thirdai::bolt {

const std::string TEMP_FILENAME = "tempTabularFile.csv";

class TabularClassifierTestFixture : public testing::Test {
 public:
  void SetUp() override {}

  void TearDown() override { std::remove(TEMP_FILENAME.c_str()); }

  void setTempFileContents(std::vector<std::string> lines) {
    std::ofstream file = dataset::SafeFileIO::ofstream(TEMP_FILENAME);
    for (auto line : lines) {
      file << line << "\n";
    }
    file.close();
  }
};

/**
 * This test asserts a failure when the user calls predict(..) before train(..).
 */
TEST_F(TabularClassifierTestFixture, TestPredictBeforeTrain) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", 2);

  std::vector<std::string> contents = {"column1,column2", "value1,value2"};
  setTempFileContents(contents);

  ASSERT_THROW(tab_model->predict(TEMP_FILENAME, std::nullopt),
               std::invalid_argument);
}

/**
 * This test asserts a failure when there is a mismatch between the number of
 * columns in the CSV provided and in the column_datatypes.
 */
TEST_F(TabularClassifierTestFixture, TestProvidedColumnsMatchCsvColumns) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", 2);
  std::vector<std::string> contents = {"column1,column2", "value1,value2"};
  setTempFileContents(contents);
  std::vector<std::string> column_datatypes = {"label"};
  ASSERT_THROW(tab_model->train(TEMP_FILENAME, column_datatypes, 1, 0.01),
               std::invalid_argument);
}

/**
 * This test asserts a failure when rows in the train/test CSVs have
 * inconsistent columns in the header (in number and in name).
 */
TEST_F(TabularClassifierTestFixture, TestTrainVSTestColumns) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", 2);
  std::vector<std::string> train_contents = {"column1,column2",
                                             "value1,value2"};
  setTempFileContents(train_contents);
  std::vector<std::string> column_datatypes = {"categorical", "label"};
  tab_model->train(TEMP_FILENAME, column_datatypes, 1, 0.01);

  std::vector<std::string> test_contents1 = {"column1,column3",
                                             "value1,value2"};
  setTempFileContents(test_contents1);
  ASSERT_THROW(tab_model->predict(TEMP_FILENAME, std::nullopt),
               std::invalid_argument);

  std::vector<std::string> test_contents2 = {"column1", "value1,value2"};
  setTempFileContents(test_contents2);
  ASSERT_THROW(tab_model->predict(TEMP_FILENAME, std::nullopt),
               std::invalid_argument);
}

// /**
//  * This test asserts a failure when a column specified as "numeric" cannot be
//  * interpreted as "numeric".
//  */
// TEST_F(TabularClassifierTestFixture, TestName) {}

// /**
//  * This test asserts a failure when a new category/label is found in the
//  testing
//  * dataset.
//  */
// TEST_F(TabularClassifierTestFixture, TestName) {}

// /**
//  * This test asserts a failure when the user forgets to specify a label
//  datatype
//  * in column_datatypes.
//  */
// TEST_F(TabularClassifierTestFixture, TestName) {}

// /**
//  * This test asserts a failure when the user specifies two label datatypes
//  * in column_datatypes.
//  */
// TEST_F(TabularClassifierTestFixture, TestName) {}

// /**
//  * This test asserts no failures when odd but valid values are found in a
//  * "numeric" column.
//  */
// TEST_F(TabularClassifierTestFixture, TestName) {}

}  // namespace thirdai::bolt
