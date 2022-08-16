#include "AutoClassifierTestUtils.h"
#include <bolt/src/auto_classifiers/WayfairClassifier.h>
#include <bolt/src/layers/BoltVector.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>

namespace thirdai::bolt::tests {

TEST(WayfairClassifierTest, TestLoadSave) {
  std::shared_ptr<bolt::WayfairClassifier> model =
      std::make_shared<WayfairClassifier>(/* n_classes= */ 5);

  std::vector<std::string> train_contents = {
      "1\t1 1", "2\t2 2",
      "3\t3 3", "4\t4 4"};

  std::vector<std::vector<uint32_t>> inference_samples = {
      {1, 1}, {2, 2},
      {3, 3}, {4, 4}};

  std::vector<std::string> single_labels = {"1", "2", "3", "4"};
  
  const std::string TRAIN_FILENAME = "tempTrainFile.csv";
  AutoClassifierTestUtils::setTempFileContents(TRAIN_FILENAME, train_contents);
  
  std::vector<float> fmeasure_thresholds = {0.9};

  model->train(TRAIN_FILENAME, /* epochs = */ 3,
               /* learning_rate = */ 0.01, 
               /* fmeasure_thresholds = */ fmeasure_thresholds);

  std::string PREDICTION_FILENAME = "predictions.csv";
  model->predict(
      /* filename = */ TRAIN_FILENAME,
      /* fmeasure_thresholds = */ fmeasure_thresholds,
      /* output_filename = */ PREDICTION_FILENAME);

  float before_load_save_accuracy =
      AutoClassifierTestUtils::computePredictFileAccuracy(PREDICTION_FILENAME,
                                                          single_labels);

  std::string SAVE_LOCATION = "textSaveLocation";
  model->save(SAVE_LOCATION);
  auto new_model = WayfairClassifier::load(SAVE_LOCATION);


  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      new_model->predict(
          /* filename = */ TRAIN_FILENAME,
          /* fmeasure_thresholds = */ fmeasure_thresholds,
          /* output_filename = */ PREDICTION_FILENAME));

  float after_load_save_accuracy =
      AutoClassifierTestUtils::computePredictFileAccuracy(PREDICTION_FILENAME,
                                                          single_labels);

  ASSERT_EQ(before_load_save_accuracy, after_load_save_accuracy);

  std::remove(TRAIN_FILENAME.c_str());
  std::remove(PREDICTION_FILENAME.c_str());
  std::remove(SAVE_LOCATION.c_str());
}

TEST(WayfairClassifierTest, TestPredictSingle) {
  std::shared_ptr<bolt::WayfairClassifier> model =
      std::make_shared<WayfairClassifier>(/* n_classes= */ 3);

  std::vector<std::string> train_contents = {
      "0,1\t1 1", "2\t2 2",
      "0,1\t1 1", "2\t2 2"};
  const std::string TRAIN_FILENAME = "tempTrainFile.csv";
  AutoClassifierTestUtils::setTempFileContents(TRAIN_FILENAME, train_contents);

  std::vector<float> fmeasure_thresholds = {0.9};

  model->train(TRAIN_FILENAME, /* epochs = */ 3,
                    /* learning_rate = */ 0.01,
                    /* fmeasure_thresholds= */ fmeasure_thresholds);

  auto output = model->predictSingle({1, 1});
  std::cout << output << std::endl;
  ASSERT_GT(output.activations[0], output.activations[2]);
  ASSERT_GT(output.activations[1], output.activations[2]);
}

TEST(WayfairClassifierTest, PredictSingleReturnsAtLeastOneActivationAboveThreshold) {
  std::shared_ptr<bolt::WayfairClassifier> model =
      std::make_shared<WayfairClassifier>(/* n_classes= */ 100);

  // Intentionally predict before training so we can expect all classes to have an original activation of < 0.9
  auto output = model->predictSingle({1, 1}, /* threshold= */ 0.0);

  float max_act = -std::numeric_limits<float>::max();
  for (uint32_t pos = 0; pos < output.len; pos++) {
    max_act = std::max(max_act, output.activations[pos]);
  }
  
  float threshold = max_act + 0.1;
  auto thresholded_output = model->predictSingle({1, 1}, threshold);

  uint32_t n_above_threshold = 0;
  for (uint32_t pos = 0; pos < output.len; pos++) {
    if (thresholded_output.activations[pos] >= threshold) {
      n_above_threshold++;
    } else {
      ASSERT_EQ(thresholded_output.activations[pos], output.activations[pos]);
    }
  }
  ASSERT_EQ(n_above_threshold, 1);

}

}  // namespace thirdai::bolt::tests