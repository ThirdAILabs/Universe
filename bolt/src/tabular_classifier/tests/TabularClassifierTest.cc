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

TEST_F(TabularClassifierTestFixture, TestOddCsvFormatsWithTabularClassifier) {
  std::cout << "Starting Tabular Tests" << std::endl;

  bolt::TabularClassifier tc("small", 2);

  std::string train_filename =
      "/Users/david/Documents/python_/tabular_experiments/data/"
      "CensusIncome/"
      "Train.csv";
  //   census income column types
  std::vector<std::string> column_datatypes = {
      "numeric",     "categorical", "numeric",     "categorical", "numeric",
      "categorical", "categorical", "categorical", "categorical", "categorical",
      "numeric",     "numeric",     "numeric",     "categorical", "label"};
  // poker hand induction column types
  //   std::vector<std::string> column_datatypes = {
  //       "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
  //       "numeric", "numeric", "numeric", "numeric", "label"};

  // OttoGroupProductClassificationChallenge columns
  //   std::vector<std::string> column_datatypes;
  //   for (int i = 0; i < 94; i++) {
  //     column_datatypes.push_back("numeric");
  //   }
  //   column_datatypes.push_back("label");

  tc.train(train_filename, column_datatypes, 1, 0.01);

  std::string test_filename =
      "/Users/david/Documents/python_/tabular_experiments/data/"
      "CensusIncome/"
      "Test.csv";

  tc.predict(test_filename, std::nullopt);
}

/**
 * This test asserts a failure when the user calls predict(..) before train(..).
 */
TEST_F(TabularClassifierTestFixture, TestPredictBeforeTrain) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", 2);

  setTempFileContents()

      tab_model->predict();
}

/**
 * This test asserts a failure when there is a mismatch between the number of
 * columns in the CSV provided and in the column_datatypes.
 */
TEST_F(TabularClassifierTestFixture, TestProvidedColumnsMatchCsvColumns) {}

/**
 * This test asserts a failure when rows in the train/test CSVs have an
 * incorrect number of columns.
 */
TEST_F(TabularClassifierTestFixture, TestName) {}

/**
 * This test asserts a failure when a column specified as "numeric" cannot be
 * interpreted as "numeric".
 */
TEST_F(TabularClassifierTestFixture, TestName) {}

/**
 * This test asserts a failure when a new category/label is found in the testing
 * dataset.
 */
TEST_F(TabularClassifierTestFixture, TestName) {}

/**
 * This test asserts a failure when the user forgets to specify a label datatype
 * in column_datatypes.
 */
TEST_F(TabularClassifierTestFixture, TestName) {}

/**
 * This test asserts a failure when the user specifies two label datatypes
 * in column_datatypes.
 */
TEST_F(TabularClassifierTestFixture, TestName) {}

/**
 * This test asserts no failures when odd but valid values are found in a
 * "numeric" column.
 */
TEST_F(TabularClassifierTestFixture, TestName) {}

}  // namespace thirdai::bolt
