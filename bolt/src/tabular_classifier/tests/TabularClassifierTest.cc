#include <bolt/src/tabular_classifier/TabularClassifier.h>
#include <gtest/gtest.h>

namespace thirdai::bolt {

class TabularClassifierTestFixture : public testing::Test {
 public:
  // define static methods here
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

// TODO TESTS
//  test for different number of columns in train vs test, in dtypes vs train,

// text random csv row has different number of columns

// test csv row has a differing datatype than specified

// test finding a new category/label in testing dataset

// test not specifing a label datatype

// test specifiing 2 label columns

// test empty csv

}  // namespace thirdai::bolt
