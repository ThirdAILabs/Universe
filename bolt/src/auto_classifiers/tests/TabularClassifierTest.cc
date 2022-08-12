#include "AutoClassifierTestUtils.h"
#include <bolt/src/auto_classifiers/TabularClassifier.h>
#include <gtest/gtest.h>

namespace thirdai::bolt {

const std::string TEMP_FILENAME = "tempTabularFile.csv";

class TabularClassifierTestFixture : public testing::Test {
 public:
  void TearDown() override { std::remove(TEMP_FILENAME.c_str()); }
};

/**
 * This test asserts a failure when the user calls predict(..) before train(..).
 */
TEST_F(TabularClassifierTestFixture, TestPredictBeforeTrain) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", /* n_classes = */ 1);

  std::vector<std::string> contents = {"colname1,colname2", "value1,label1",
                                       "value3,label2"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, contents);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      tab_model->predict(/* filename = */ TEMP_FILENAME,
                         /* output_filename = */ std::nullopt),
      std::invalid_argument);
}

/**
 * This test asserts a failure when there is a mismatch between the number of
 * columns in the CSV provided and in the column_datatypes.
 */
TEST_F(TabularClassifierTestFixture, TestProvidedColumnsMatchCsvColumns) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", /* n_classes = */ 1);
  std::vector<std::string> contents = {"colname1,colname2", "value1,label1",
                                       "value3,label2"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, contents);
  std::vector<std::string> column_datatypes = {"label"};
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      tab_model->train(TEMP_FILENAME, column_datatypes, /* epochs = */ 1,
                       /* learning_rate = */ 0.01),
      std::invalid_argument);
}

/**
 * This test asserts a failure when rows in the train/test CSVs have
 * inconsistent columns.
 */
TEST_F(TabularClassifierTestFixture, TestTrainVSTestColumns) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", /* n_classes = */ 2);
  std::vector<std::string> train_contents = {"colname1,colname2",
                                             "value1,label1", "value3,label2"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, train_contents);
  std::vector<std::string> column_datatypes = {"categorical", "label"};
  tab_model->train(TEMP_FILENAME, column_datatypes, /* epochs = */ 1,
                   /* learning_rate = */ 0.01);

  std::vector<std::string> test_contents = {"colname1,colname2", "value1",
                                            "value3,label2"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, test_contents);
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      tab_model->predict(/* filename = */ TEMP_FILENAME,
                         /* output_filename = */ std::nullopt),
      std::invalid_argument);
}

/**
 * This test asserts a failure when a column specified as "numeric" cannot be
 * interpreted as "numeric".
 */
TEST_F(TabularClassifierTestFixture, TestIncorrectNumericColumn) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", /* n_classes = */ 1);
  std::vector<std::string> contents = {"colname1,colname2", "value1,label1",
                                       "value3,label2"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, contents);
  std::vector<std::string> column_datatypes = {"numeric", "label"};
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      tab_model->train(TEMP_FILENAME, column_datatypes, /* epochs = */ 1,
                       /* learning_rate = */ 0.01),
      std::invalid_argument);
}

/**
 * This test asserts no failure when an empty value is passed in to categorical
 * or numeric columns.
 */
TEST_F(TabularClassifierTestFixture, TestEmptyColumns) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", /* n_classes = */ 2);
  std::vector<std::string> contents = {"colname1,colname2,colname3,colname4",
                                       "value1,2,value3, label1",
                                       "value1,,,label2"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, contents);
  std::vector<std::string> column_datatypes = {"categorical", "numeric",
                                               "categorical", "label"};

  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      tab_model->train(TEMP_FILENAME, column_datatypes, /* epochs = */ 1,
                       /* learning_rate = */ 0.01));

  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      tab_model->predict(/* filename = */ TEMP_FILENAME,
                         /* output_filename = */ std::nullopt));
}

/**
 * This test asserts a failure when a new category/label is found in the testing
 * dataset.
 */
TEST_F(TabularClassifierTestFixture, TestFailureOnNewTestLabel) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", /* n_classes = */ 2);
  std::vector<std::string> train_contents = {"colname1,colname2",
                                             "value1,label1", "value2,label2"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, train_contents);
  std::vector<std::string> column_datatypes = {"categorical", "label"};
  tab_model->train(TEMP_FILENAME, column_datatypes, /* epochs = */ 1,
                   /* learning_rate = */ 0.01);

  std::vector<std::string> test_contents = {"colname1,colname2",
                                            "value1,label1", "value2,label99"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, test_contents);
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      tab_model->predict(/* filename = */ TEMP_FILENAME,
                         /* output_filename = */ std::nullopt),
      std::invalid_argument);
}

/**
 * This test asserts a failure when more labels are in the dataset than
 * specified in the constructor
 */
TEST_F(TabularClassifierTestFixture, TestTooManyLabels) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", /* n_classes = */ 1);
  std::vector<std::string> train_contents = {"colname1,colname2",
                                             "value1,label1", "value2,label2"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, train_contents);
  std::vector<std::string> column_datatypes = {"categorical", "label"};
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      tab_model->train(TEMP_FILENAME, column_datatypes, /* epochs = */ 1,
                       /* learning_rate = */ 0.01),
      std::invalid_argument);
}

/**
 * This test asserts a failure when the user forgets to specify a label datatype
 * in column_datatypes.
 */
TEST_F(TabularClassifierTestFixture, TestNoLabelDatatype) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", /* n_classes = */ 1);
  std::vector<std::string> train_contents = {"colname1,colname2",
                                             "value1,label1", "value2,label2"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, train_contents);
  std::vector<std::string> column_datatypes = {"categorical", "numeric"};
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      tab_model->train(TEMP_FILENAME, column_datatypes, /* epochs = */ 1,
                       /* learning_rate = */ 0.01),
      std::invalid_argument);
}

/**
 * This test asserts a failure when the user specifies two label datatypes
 * in column_datatypes.
 */
TEST_F(TabularClassifierTestFixture, TestFailureOnTwoLabelColumns) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", /* n_classes = */ 1);
  std::vector<std::string> train_contents = {"colname1,colname2",
                                             "value1,label1", "value2,label2"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, train_contents);
  std::vector<std::string> column_datatypes = {"label", "label"};
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      tab_model->train(TEMP_FILENAME, column_datatypes, /* epochs = */ 1,
                       /* learning_rate = */ 0.01),
      std::invalid_argument);
}

TEST_F(TabularClassifierTestFixture, TestLoadSave) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", 4);

  std::vector<std::string> train_contents = {
      "colname1,colname2,colname3", "value1,value1,label1",
      "value2,value2,label2", "value3,value3,label3", "value4,value4,label4"};
  std::vector<std::string> labels = {"label1", "label2", "label3", "label4"};
  AutoClassifierTestUtils::setTempFileContents(TEMP_FILENAME, train_contents);

  std::vector<std::string> column_datatypes = {"categorical", "categorical",
                                               "label"};

  tab_model->train(TEMP_FILENAME, column_datatypes, /* epochs = */ 3,
                   /* learning_rate = */ 0.01);

  std::string PREDICTION_FILENAME = "predictions.csv";
  tab_model->predict(
      /* filename = */ TEMP_FILENAME,
      /* output_filename = */ PREDICTION_FILENAME);

  float before_load_save_accuracy =
      AutoClassifierTestUtils::computePredictFileAccuracy(PREDICTION_FILENAME,
                                                          labels);

  std::string SAVE_LOCATION = "tabularSaveLocation";
  tab_model->save(SAVE_LOCATION);
  auto new_model = TabularClassifier::load(SAVE_LOCATION);

  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      new_model->predict(
          /* filename = */ TEMP_FILENAME,
          /* output_filename = */ PREDICTION_FILENAME));

  float after_load_save_accuracy =
      AutoClassifierTestUtils::computePredictFileAccuracy(PREDICTION_FILENAME,
                                                          labels);

  ASSERT_EQ(before_load_save_accuracy, after_load_save_accuracy);

  std::remove(PREDICTION_FILENAME.c_str());
  std::remove(SAVE_LOCATION.c_str());
}

TEST_F(TabularClassifierTestFixture, TestPredictSingle) {
  std::shared_ptr<bolt::TabularClassifier> tab_model =
      std::make_shared<TabularClassifier>("small", 2);

  std::vector<std::string> train_contents = {
      "colname1,colname2,colname3", "value1,value1,label1",
      "value2,value2,label2", "value1,value1,label1", "value2,value2,label2"};
  const std::string TRAIN_FILENAME = "tempTrainFile.csv";
  AutoClassifierTestUtils::setTempFileContents(TRAIN_FILENAME, train_contents);

  std::vector<std::string> column_datatypes = {"categorical", "categorical",
                                               "label"};

  tab_model->train(TRAIN_FILENAME, column_datatypes, /* epochs = */ 3,
                   /* learning_rate = */ 0.01);

  std::vector<std::string> sample = {"value1", "value1"};
  ASSERT_EQ(tab_model->predictSingle(sample), "label1");
}

}  // namespace thirdai::bolt
