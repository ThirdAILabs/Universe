#include "AutoClassifierTestUtils.h"
#include <bolt/src/auto_classifiers/TextClassifier.h>
#include <gtest/gtest.h>
#include <fstream>

namespace thirdai::bolt::tests {

TEST(TextClassifierTest, TestLoadSave) {
  std::shared_ptr<bolt::TextClassifier> text_model =
      std::make_shared<TextClassifier>("small", 4);

  std::vector<std::string> train_contents = {
      "text, category", "value1 value1,label1", "value2 value2,label2",
      "value3 value3,label3", "value4 value4,label4"};
  std::vector<std::string> labels = {"label1", "label2", "label3", "label4"};
  const std::string TRAIN_FILENAME = "tempTrainFile.csv";
  AutoClassifierTestUtils::setTempFileContents(TRAIN_FILENAME, train_contents);

  text_model->train(TRAIN_FILENAME, /* epochs = */ 3,
                    /* learning_rate = */ 0.01);

  std::string PREDICTION_FILENAME = "predictions.csv";
  text_model->predict(
      /* filename = */ TRAIN_FILENAME,
      /* output_filename = */ PREDICTION_FILENAME);

  float before_load_save_accuracy =
      AutoClassifierTestUtils::computePredictFileAccuracy(PREDICTION_FILENAME,
                                                          labels);

  std::string SAVE_LOCATION = "textSaveLocation";
  text_model->save(SAVE_LOCATION);
  auto new_model = TextClassifier::load(SAVE_LOCATION);

  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      new_model->predict(
          /* filename = */ TRAIN_FILENAME,
          /* output_filename = */ PREDICTION_FILENAME));

  float after_load_save_accuracy =
      AutoClassifierTestUtils::computePredictFileAccuracy(PREDICTION_FILENAME,
                                                          labels);

  ASSERT_EQ(before_load_save_accuracy, after_load_save_accuracy);

  std::remove(TRAIN_FILENAME.c_str());
  std::remove(PREDICTION_FILENAME.c_str());
  std::remove(SAVE_LOCATION.c_str());
}

TEST(TextClassifierTest, TestPredictSingle) {
  std::shared_ptr<bolt::TextClassifier> text_model =
      std::make_shared<TextClassifier>("small", 2);

  std::vector<std::string> train_contents = {
      "text, category", "value1 value1,label1", "value2 value2,label2",
      "value1 value1,label1", "value2 value2,label2"};
  const std::string TRAIN_FILENAME = "tempTrainFile.csv";
  AutoClassifierTestUtils::setTempFileContents(TRAIN_FILENAME, train_contents);

  text_model->train(TRAIN_FILENAME, /* epochs = */ 3,
                    /* learning_rate = */ 0.01);

  ASSERT_EQ(text_model->predictSingle("value1 value1"), "label1");
}

}  // namespace thirdai::bolt::tests