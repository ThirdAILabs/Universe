#include <bolt/src/auto_classifiers/sequential_classifier/ConstructorUtilityTypes.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialClassifier.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialUtils.h>
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <dataset/src/utils/TimeUtils.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <optional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::bolt::sequential_classifier::tests {

constexpr const char* TRAIN_FILE_NAME = "seq_class_train.csv";
constexpr const char* TEST_FILE_NAME = "seq_class_test.csv";
constexpr const char* MODEL_SAVE_FILE_NAME = "seq_class_save";

void writeRowsToFile(const std::string& filename,
                     std::vector<std::string>&& rows) {
  std::ofstream file(filename);
  for (const auto& row : rows) {
    file << row << std::endl;
  }
}

void assertSuccessfulLoadSave(
    SequentialClassifier& model,
    const std::unordered_map<std::string, std::string>& predict_single_sample,
    uint32_t n_targets) {
  model.train(TRAIN_FILE_NAME, /* epochs= */ 5, /* learning_rate= */ 0.01);

  // Save before making original prediction so both calls to predict() use the
  // same starting data states (vocabulary and item history).
  model.save(MODEL_SAVE_FILE_NAME);
  auto loaded_model = SequentialClassifier::load(MODEL_SAVE_FILE_NAME);

  auto original_model_results =
      model.predict(TEST_FILE_NAME, /* metrics= */ {"recall@1"});
  auto original_model_single_output =
      model.predictSingle(predict_single_sample, n_targets);

  auto loaded_model_results =
      loaded_model->predict(TEST_FILE_NAME, /* metrics= */ {"recall@1"});
  auto loaded_model_single_output =
      loaded_model->predictSingle(predict_single_sample, n_targets);

  ASSERT_EQ(original_model_results["recall@1"],
            loaded_model_results["recall@1"]);

  std::remove(TRAIN_FILE_NAME);
  std::remove(TEST_FILE_NAME);
  std::remove(MODEL_SAVE_FILE_NAME);

  ASSERT_EQ(original_model_single_output.size(),
            loaded_model_single_output.size());
  for (uint32_t i = 0; i < original_model_single_output.size(); i++) {
    auto& [original_class_name, original_activation] =
        original_model_single_output[i];
    auto& [loaded_class_name, loaded_activation] =
        loaded_model_single_output[i];
    ASSERT_EQ(original_class_name, loaded_class_name);
    ASSERT_EQ(original_activation, loaded_activation);
  }
}

void assertFailsTraining(SequentialClassifier& model) {
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      model.train(TRAIN_FILE_NAME, /* epochs= */ 5, /* learning_rate= */ 0.01),
      std::invalid_argument);

  std::remove(TRAIN_FILE_NAME);
}

SequentialClassifier getTrainedClassifier(const char* train_file_name) {
  SequentialClassifier classifier(
      /* data_types= */
      {
          {"user", DataType::categorical(/* n_unique_classes= */ 1)},
          {"target", DataType::categorical(/* n_unique_classes= */ 2)},
          {"timestamp", DataType::date()},
          {"static_text", DataType::text()},
          {"static_category", DataType::categorical(/* n_unique_classes= */ 4)},
          {"sequential", DataType::categorical(/* n_unique_classes= */ 2)},
      },
      /* temporal_tracking_relationships= */
      {{"user",
        {TemporalConfig::categorical(/* column_name= */ "sequential",
                                     /* track_last_n= */ 3)}}},
      /* target= */ "target");

  classifier.train(train_file_name, /* epochs= */ 5, /* learning_rate= */ 0.01);

  return classifier;
}

std::vector<std::string> getWordsInTextColumn(const std::string& sentence) {
  std::vector<std::string> text_reasons;
  std::string token;
  std::stringstream ss(sentence);
  while (getline(ss, token, ' ')) {
    text_reasons.push_back(token);
  }
  return text_reasons;
}

void assertColumnNames(std::vector<std::string> column_names,
                       std::unordered_map<std::string, std::string> input) {
  // here we should have 'timestamp' four times because we are using four values
  // in input from the timestamp. we should have 'sequential' three times
  // because we are tracking last three values in the schema. for remaining
  // columns we should only have 1 because we are making that from categorical.
  auto copy_column_names = column_names;

  auto iter = std::unique(copy_column_names.begin(), copy_column_names.end());

  copy_column_names.resize(std::distance(copy_column_names.begin(), iter));

  for (const auto& column_name : copy_column_names) {
    if (column_name == "timestamp") {
      ASSERT_EQ(
          std::count(column_names.begin(), column_names.end(), column_name), 4);
    } else if (column_name == "sequential") {
      ASSERT_LE(
          std::count(column_names.begin(), column_names.end(), column_name), 3);
    } else if (column_name == "static_text") {
      std::vector<std::string> text_reasons =
          getWordsInTextColumn(input["static_text"]);
      ASSERT_EQ(
          std::count(column_names.begin(), column_names.end(), column_name),
          text_reasons.size());
    } else {
      ASSERT_EQ(
          std::count(column_names.begin(), column_names.end(), column_name), 1);
    }
  }
}

void assertPercentageSignificance(std::vector<float> percentage_significances) {
  // assert the values are sorted in descending order of absolute values.
  bool isSorted = std::is_sorted(percentage_significances.begin(),
                                 percentage_significances.end(),
                                 [](float value1, float value2) {
                                   return std::abs(value1) > std::abs(value2);
                                 });

  ASSERT_TRUE(isSorted);

  // assert the total sum of absolute values is close to 100.
  float total_percentage_sum = 0.0;

  for (auto percentage_significance : percentage_significances) {
    total_percentage_sum += std::abs(percentage_significance);
  }

  ASSERT_GT(total_percentage_sum, 99.9);
}

void assertWordsWithinBlock(const std::vector<std::string>& column_names,
                            std::unordered_map<std::string, std::string> input,
                            const std::vector<std::string>& words_responsible) {
  std::vector<std::string> timestamp_reasons = {
      "day_of_week", "week_of_month", "month_of_year", "week_of_year"};
  // these sequential reasons based on values in the sequential column in train
  // data.
  std::vector<std::string> sequential_reasons = {
      "Previously seen 'A'", "Previously seen 'B'", "Previously seen 'C'",
      "Previously seen 'D'"};
  std::vector<std::string> text_reasons =
      getWordsInTextColumn(input["static_text"]);
  for (uint32_t i = 0; i < words_responsible.size(); i++) {
    if (column_names[i] == "timestamp") {
      ASSERT_TRUE(std::find(timestamp_reasons.begin(), timestamp_reasons.end(),
                            words_responsible[i]) != timestamp_reasons.end());
    } else if (column_names[i] == "sequential") {
      ASSERT_TRUE(std::find(sequential_reasons.begin(),
                            sequential_reasons.end(),
                            words_responsible[i]) != sequential_reasons.end());
    } else if (column_names[i] == "static_text") {
      ASSERT_TRUE(std::find(text_reasons.begin(), text_reasons.end(),
                            words_responsible[i]) != text_reasons.end());
    } else {
      ASSERT_TRUE(input[column_names[i]] == words_responsible[i]);
    }
  }
}

TEST(SequentialClassifierTest, TestLoadSave) {
  writeRowsToFile(TRAIN_FILE_NAME,
                  {"user,target,timestamp,static_text,static_category,count",
                   "0,0,2022-08-29,hello,0,0", "0,1,2022-08-30,hello,1,1",
                   "0,0,2022-08-31,hello,2,2", "0,1,2022-09-01,hello,3,3"});

  writeRowsToFile(TEST_FILE_NAME,
                  {"user,target,timestamp,static_text,static_category,count",
                   "0,0,2022-09-02,hello,0,4", "0,1,2022-09-03,hello,1,5",
                   "0,0,2022-09-04,hello,2,6", "0,1,2022-09-05,hello,3,7"});

  std::unordered_map<std::string, std::string> predict_single_sample = {
      {"user", "0"},
      {"target", "0"},
      {"timestamp", "2022-09-06"},
      {"static_text", "hello"},
      {"static_category", "0"},
      {"count", "8"},
  };

  SequentialClassifier model(
      /* data_types= */
      {
          {"user", DataType::categorical(/* n_unique_classes= */ 1)},
          {"target", DataType::categorical(/* n_unique_classes= */ 2)},
          {"timestamp", DataType::date()},
          {"static_text", DataType::text()},
          {"static_category", DataType::categorical(/* n_unique_classes= */ 4)},
          {"count", DataType::numerical(std::make_pair(1, 7))},
      },
      /* temporal_tracking_relationships= */
      {{"user",
        {
            TemporalConfig::categorical(/* column_name= */ "target",
                                        /* track_last_n= */ 3),
            TemporalConfig::numerical(/* column_name= */ "count",
                                      /* history_length= */ 5),
        }}},
      /* target= */ "target",
      /* time_granularity= */ "daily",
      /* lookahead= */ 1);

  assertSuccessfulLoadSave(model, predict_single_sample, /* n_targets= */ 2);
}

TEST(SequentialClassifierTest, TestExplainMethod) {
  writeRowsToFile(
      TRAIN_FILE_NAME,
      {"user,target,timestamp,static_text,static_category,sequential",
       "0,0,2022-08-29,hello,0,B", "0,1,2022-08-30,hello,1,A",
       "0,0,2022-08-31,hello,2,A", "0,1,2022-09-01,hello,3,B"});

  SequentialClassifier classifier = getTrainedClassifier(TRAIN_FILE_NAME);

  std::unordered_map<std::string, std::string> single_inference_input = {
      {"user", "0"},
      {"target", "0"},
      {"timestamp", "2022-09-01"},
      {"static_text", "hello world"},
      {"static_category", "0"},
      {"sequential", "B"}};

  std::vector<dataset::Explanation> responsible_column_and_input_keys =
      classifier.explain(single_inference_input);

  std::vector<std::string> column_names;
  std::vector<std::string> words_responsible;
  std::vector<float> percentage_significances;

  for (auto& responsible_column_and_input_key :
       responsible_column_and_input_keys) {
    percentage_significances.push_back(
        responsible_column_and_input_key.percentage_significance);
    column_names.push_back(responsible_column_and_input_key.column_name);
    words_responsible.push_back(responsible_column_and_input_key.keyword);
  }

  // we will check how many times the column names are present in the vector.
  assertColumnNames(column_names, single_inference_input);

  // we will check the total percentage is close to 100 and the percentage
  // significance are sorted.

  assertPercentageSignificance(percentage_significances);

  // we will check the words responsible are there in the input or not.
  assertWordsWithinBlock(column_names, single_inference_input,
                         words_responsible);

  std::remove(TRAIN_FILE_NAME);
}

TEST(SequentialClassifierTest, TestDenseSequentialFeatures) {
  SequentialClassifier model(
      /* data_types= */
      {
          {"user", DataType::categorical(/* n_unique_classes= */ 1)},
          {"target", DataType::categorical(/* n_unique_classes= */ 2)},
          {"timestamp", DataType::date()},
          {"count", DataType::numerical(std::make_pair(1, 7))},
      },
      /* temporal_tracking_relationships= */
      {{"user",
        {
            TemporalConfig::numerical(/* column_name= */ "count",
                                      /* history_length= */ 5),
            TemporalConfig::categorical(/* column_name= */ "target",
                                        /* track_last_n= */ 3),
        }}},
      /* target= */ "target",
      /* time_granularity= */ "biweekly",
      /* lookahead= */ 1);

  /*
    Train before getting state because state is only built once
    we call .train(), .predict(), or .explain()
  */
  writeRowsToFile(TRAIN_FILE_NAME, {"user,target,timestamp,count",
                                    "0,0,2022-09-04,6", "0,1,2022-09-05,7"});
  model.train(TRAIN_FILE_NAME, /* epochs= */ 1, /* learning_rate= */ 0.01);

  auto state = SequentialClassifierTextFixture::getState(model);

  auto count_history = state.quantity_histories_by_id["user_0"];
  ASSERT_EQ(count_history->historyLag(), 1);
  ASSERT_EQ(count_history->historyLength(), 5);
  ASSERT_EQ(count_history->granularity(),
            dataset::QuantityTrackingGranularity::Biweekly);
}

}  // namespace thirdai::bolt::sequential_classifier::tests