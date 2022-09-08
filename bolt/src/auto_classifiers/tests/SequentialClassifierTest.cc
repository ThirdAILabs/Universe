#include <bolt/src/auto_classifiers/sequential_classifier/SequentialClassifier.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialUtils.h>
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace thirdai::bolt::sequential_classifier::tests {

void writeMockSequentialDataToFile(
    const std::string& train_file_name,
    const std::optional<std::string>& test_file_name = std::nullopt) {
  std::string header =
      "user,target,timestamp,static_text,static_categorical,sequential";

  std::ofstream train_file(train_file_name);
  train_file << header << std::endl;
  train_file << "0,0,2022-08-29,hello,0,A" << std::endl;
  train_file << "0,1,2022-08-30,hello,1,B" << std::endl;
  train_file << "0,0,2022-08-31,hello,2,C" << std::endl;
  train_file << "0,1,2022-09-01,hello,3,D" << std::endl;

  if (test_file_name != std::nullopt) {
    std::ofstream test_file(*test_file_name);
    test_file << header << std::endl;
    test_file << "0,0,2022-09-02,hello,0,E" << std::endl;
    test_file << "0,1,2022-09-03,hello,1,B" << std::endl;
    test_file << "0,0,2022-09-04,hello,2,C" << std::endl;
    test_file << "0,1,2022-09-05,hello,3,D" << std::endl;
  }
}

SequentialClassifier makeSequentialClassifierForMockData() {
  CategoricalPair user = {"user", 5};
  CategoricalPair target = {"target", 5};
  std::string timestamp = "timestamp";
  std::vector<std::string> static_text = {"static_text"};
  std::vector<CategoricalPair> static_categorical = {{"static_categorical", 5}};
  std::vector<SequentialTriplet> sequential = {{"sequential", 5, 3}};
  return {user, target, timestamp, static_text, static_categorical, sequential};
}

std::unordered_map<std::string, std::string>
mockSequentialSampleForPredictSingle() {
  return {{"user", "0"},
          {"target", "0"},
          {"timestamp", "2022-09-01"},
          {"static_text", "hello world"},
          {"static_categorical", "0"},
          {"sequential", "B"}};
}

SequentialClassifier getTrainedClassifier(const char* train_file_name) {
  auto classifier = makeSequentialClassifierForMockData();

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

void assert_column_names(std::vector<std::string> column_names,
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
      ASSERT_EQ(
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

void assert_percentage_significance(
    std::vector<float> percentage_significances) {
  // assert the values are sorted in descending order of absolute values.
  bool isSorted = std::is_sorted(percentage_significances.begin(),
                                 percentage_significances.end(),
                                 [](float value1, float value2) {
                                   return std::abs(value1) >= std::abs(value2);
                                 });

  ASSERT_TRUE(isSorted);

  // assert the total sum of absolute values is close to 100.
  float total_percentage_sum = 0.0;

  for (auto percentage_significance : percentage_significances) {
    total_percentage_sum += std::abs(percentage_significance);
  }

  ASSERT_GT(total_percentage_sum, 0.99);
}

void assert_words_within_block(
    const std::vector<std::string>& column_names,
    std::unordered_map<std::string, std::string> input,
    const std::vector<std::string>& words_responsible) {
  std::vector<std::string> timestamp_reasons = {
      "day_of_week", "week_of_month", "month_of_year", "week_of_year"};
  // these sequential reasons based on values in the sequential column in train
  // data.
  std::vector<std::string> sequential_reasons = {"A", "B", "C", "D"};
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
  const char* train_file_name = "seq_class_train.csv";
  const char* test_file_name = "seq_class_test.csv";
  const char* model_save_file_name = "seq_class_save";

  writeMockSequentialDataToFile(train_file_name, test_file_name);

  auto classifier = getTrainedClassifier(train_file_name);

  // Save before making original prediction so both calls to predict() use the
  // same starting data states (vocabulary and item history).
  classifier.save(model_save_file_name);
  auto new_classifier = SequentialClassifier::load(model_save_file_name);

  auto original_model_results =
      classifier.predict(test_file_name, /* metrics= */ {"recall@1"});

  auto loaded_model_results =
      new_classifier->predict(test_file_name, /* metrics= */ {"recall@1"});

  ASSERT_EQ(original_model_results.first["recall@1"],
            loaded_model_results.first["recall@1"]);

  std::remove(train_file_name);
  std::remove(test_file_name);
  std::remove(model_save_file_name);
}

TEST(SequentialClassifierTest, TestExplainMethod) {
  const char* train_file_name = "seq_class_train.csv";

  writeMockSequentialDataToFile(train_file_name);

  auto classifier = getTrainedClassifier(train_file_name);

  auto single_inference_input = mockSequentialSampleForPredictSingle();

  auto [column_names, percentage_significances, words_responsible] =
      classifier.explain(single_inference_input);

  // we will check how many times the column names are present in the vector.
  assert_column_names(column_names, single_inference_input);

  // we will check the total percentage is close to 100 and the percentage
  // significance are sorted.

  assert_percentage_significance(percentage_significances);

  // we will check the words responsible are there in the input or not.
  assert_words_within_block(column_names, single_inference_input,
                            words_responsible);

  std::remove(train_file_name);
}

}  // namespace thirdai::bolt::sequential_classifier::tests