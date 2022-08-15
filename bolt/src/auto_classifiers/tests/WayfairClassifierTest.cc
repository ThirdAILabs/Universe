#include "AutoClassifierTestUtils.h"
#include <bolt/src/auto_classifiers/WayfairClassifier.h>
#include <bolt/src/layers/BoltVector.h>
#include <gtest/gtest.h>
#include <fstream>
#include <memory>

namespace thirdai::bolt::tests {

std::vector<BoltVector> getPredictionOutputs(std::shared_ptr<bolt::WayfairClassifier> model, const std::vector<std::vector<uint32_t>>& samples) {
  std::vector<BoltVector> outputs;
  outputs.reserve(samples.size());
  for (const auto& tokens : samples) {
    outputs.push_back(model->predictSingle(tokens));
  }
  return outputs;
}

// checkConsistentWithPredictSingle (but this only works if we dont do sparse inference)
// getSinglePredictionAccuracies

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

  model->train(TRAIN_FILENAME, /* epochs = */ 3,
               /* learning_rate = */ 0.01, 
               /* fmeasure_threshold = */ 0.9);

  std::string PREDICTION_FILENAME = "predictions.csv";
  model->predict(
      /* filename = */ TRAIN_FILENAME,
      /* fmeasure_threshold = */ 0.9,
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
          /* fmeasure_threshold = */ 0.9,
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

  model->train(TRAIN_FILENAME, /* epochs = */ 3,
                    /* learning_rate = */ 0.01,
                    /* fmeasure_threshold= */ 0.9);

  auto output = model->predictSingle({1, 1});
  std::cout << output << std::endl;
  ASSERT_GT(output.activations[0], output.activations[2]);
  ASSERT_GT(output.activations[1], output.activations[2]);
}

}  // namespace thirdai::bolt::tests