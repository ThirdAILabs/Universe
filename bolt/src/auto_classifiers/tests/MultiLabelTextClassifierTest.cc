#include "AutoClassifierTestUtils.h"
#include <bolt/src/auto_classifiers/MultiLabelTextClassifier.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/metrics/Metric.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

namespace thirdai::bolt::tests {

float getFMeasure(std::vector<BoltVector> outputs,
                  std::vector<BoltVector> labels, float threshold) {
  FMeasure metric(threshold);
  for (uint32_t vec_idx = 0; vec_idx < outputs.size(); vec_idx++) {
    metric.computeMetric(outputs[vec_idx], labels[vec_idx]);
  }
  return metric.getMetricAndReset(/* verbose= */ false);
}

TEST(MultiLabelTextClassifierTest, TestLoadSave) {
  std::shared_ptr<MultiLabelTextClassifier> model =
      std::make_shared<MultiLabelTextClassifier>(/* n_classes= */ 5);

  std::vector<std::string> train_contents = {"1\t1 1", "2\t2 2", "3\t3 3",
                                             "4\t4 4"};

  std::vector<std::string> single_labels = {"1", "2", "3", "4"};

  const std::string TRAIN_FILENAME = "tempTrainFile.csv";
  AutoClassifierTestUtils::setTempFileContents(TRAIN_FILENAME, train_contents);

  std::vector<std::string> metrics = {"f_measure(0.9)"};

  model->train(TRAIN_FILENAME, /* epochs = */ 3,
               /* learning_rate = */ 0.01,
               /* metrics = */ metrics);

  auto [metrics_before_saving, _1] = model->predict(
      /* filename = */ TRAIN_FILENAME,
      /* metrics = */ metrics);

  std::string SAVE_LOCATION = "textSaveLocation";
  model->save(SAVE_LOCATION);
  auto new_model = MultiLabelTextClassifier::load(SAVE_LOCATION);

  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      new_model->predict(
          /* filename = */ TRAIN_FILENAME,
          /* metrics = */ metrics));

  auto [metrics_after_loading, _2] = new_model->predict(
      /* filename = */ TRAIN_FILENAME,
      /* metrics = */ metrics);

  for (auto& [key, value] : metrics_before_saving) {
    ASSERT_EQ(value, metrics_after_loading[key]);
  }

  std::remove(TRAIN_FILENAME.c_str());
  std::remove(SAVE_LOCATION.c_str());
}

TEST(MultiLabelTextClassifierTest, TestPredictSingle) {
  std::shared_ptr<bolt::MultiLabelTextClassifier> model =
      std::make_shared<MultiLabelTextClassifier>(/* n_classes= */ 3);

  std::vector<std::string> train_contents = {"0,1\t1 1", "2\t2 2", "0,1\t1 1",
                                             "2\t2 2"};
  const std::string TRAIN_FILENAME = "tempTrainFile.csv";
  AutoClassifierTestUtils::setTempFileContents(TRAIN_FILENAME, train_contents);

  std::vector<std::string> metrics = {"f_measure(0.9)"};

  model->train(TRAIN_FILENAME, /* epochs = */ 3,
               /* learning_rate = */ 0.01,
               /* metrics= */ metrics);

  auto output_from_tokens = model->predictSingleFromTokens({1, 1});
  auto output_from_sentence = model->predictSingleFromSentence("1 1");

  /*
    Since the tokens {1, 1} should map to labels 0 and 1,
    we expect that activations of classes 0 and 1 are
    both greater than the activation of class 2.
  */
  ASSERT_GT(output_from_tokens.activations[0],
            output_from_tokens.activations[2]);
  ASSERT_GT(output_from_tokens.activations[1],
            output_from_tokens.activations[2]);
  ASSERT_EQ(output_from_tokens.activations[0],
            output_from_sentence.activations[0]);
  ASSERT_EQ(output_from_tokens.activations[1],
            output_from_sentence.activations[1]);
  ASSERT_EQ(output_from_tokens.activations[2],
            output_from_sentence.activations[2]);
}

/**
 * This classifier is built for multi label classification with a
 * threshold based approach. That is, we retrieve classes whose
 * activations exceed the given threshold. If no class exceeds this
 * threshold, then the model should artificially set the highest
 * activation to this threshold so that at least one label is
 * returned.
 */
TEST(MultiLabelTextClassifierTest,
     PredictSingleReturnsAtLeastOneActivationAboveThreshold) {
  std::shared_ptr<bolt::MultiLabelTextClassifier> model =
      std::make_shared<MultiLabelTextClassifier>(/* n_classes= */ 100);

  // Intentionally predict before training so we can expect most classes
  auto output = model->predictSingleFromTokens({1, 1}, /* threshold= */ 0.0);

  float max_act = -std::numeric_limits<float>::max();
  for (uint32_t pos = 0; pos < output.len; pos++) {
    max_act = std::max(max_act, output.activations[pos]);
  }

  float threshold = max_act + 0.1;
  auto thresholded_output = model->predictSingleFromTokens({1, 1}, threshold);

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

TEST(MultiLabelTextClassifierTest, ConsistentPredictAndPredictSingle) {
  std::shared_ptr<bolt::MultiLabelTextClassifier> model =
      std::make_shared<MultiLabelTextClassifier>(/* n_classes= */ 500);

  std::vector<std::string> train_contents = {"1,10\t1 1", "2,20,200\t2 2",
                                             "3\t3 3", "4,400\t4 4"};

  const std::string TRAIN_FILENAME = "tempTrainFile.csv";
  AutoClassifierTestUtils::setTempFileContents(TRAIN_FILENAME, train_contents);

  std::vector<float> f_measure_thresholds = {0.1, 0.2, 0.9};
  std::vector<std::string> metrics = {"f_measure(0.1)", "f_measure(0.2)",
                                      "f_measure(0.9)"};

  model->train(TRAIN_FILENAME, /* epochs = */ 5,
               /* learning_rate = */ 0.01,
               /* metrics = */ metrics);

  auto [prediction_metrics, _] = model->predict(
      /* filename = */ TRAIN_FILENAME,
      /* metrics = */ metrics);

  std::vector<BoltVector> vector_labels = {
      BoltVector::makeSparseVector({1, 10}, {1.0, 1.0}),
      BoltVector::makeSparseVector({2, 20, 200}, {1.0, 1.0, 1.0}),
      BoltVector::makeSparseVector({3}, {1.0}),
      BoltVector::makeSparseVector({4, 400}, {1.0, 1.0})};

  std::vector<std::vector<uint32_t>> single_inference_token_samples = {
      {1, 1}, {2, 2}, {3, 3}, {4, 4}};
  std::vector<std::string> single_inference_sentence_samples = {"1 1", "2 2",
                                                                "3 3", "4 4"};

  std::vector<BoltVector> token_single_inference_outputs;
  std::vector<BoltVector> sentence_single_inference_outputs;
  token_single_inference_outputs.reserve(single_inference_token_samples.size());
  sentence_single_inference_outputs.reserve(
      single_inference_sentence_samples.size());

  for (auto& sample : single_inference_token_samples) {
    token_single_inference_outputs.push_back(
        model->predictSingleFromTokens(sample));
  }
  for (auto& sample : single_inference_sentence_samples) {
    sentence_single_inference_outputs.push_back(
        model->predictSingleFromSentence(sample));
  }

  for (uint32_t i = 0; i < f_measure_thresholds.size(); i++) {
    ASSERT_NEAR(getFMeasure(token_single_inference_outputs, vector_labels,
                            f_measure_thresholds[i]),
                prediction_metrics[metrics[i]],
                /* abs_error= */ 0.000001);
    ASSERT_NEAR(getFMeasure(sentence_single_inference_outputs, vector_labels,
                            f_measure_thresholds[i]),
                prediction_metrics[metrics[i]],
                /* abs_error= */ 0.000001);
  }
}

}  // namespace thirdai::bolt::tests