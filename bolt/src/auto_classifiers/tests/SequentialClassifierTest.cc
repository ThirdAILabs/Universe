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
  train_file << "0,0,2022-08-29,hello,0,1" << std::endl;
  train_file << "0,1,2022-08-30,hello,1,2" << std::endl;
  train_file << "0,0,2022-08-31,hello,2,3" << std::endl;
  train_file << "0,1,2022-09-01,hello,3,4" << std::endl;

  if (test_file_name != std::nullopt) {
    std::ofstream test_file(*test_file_name);
    test_file << header << std::endl;
    test_file << "0,0,2022-09-02,hello,0,5" << std::endl;
    test_file << "0,1,2022-09-03,hello,1,2" << std::endl;
    test_file << "0,0,2022-09-04,hello,2,3" << std::endl;
    test_file << "0,1,2022-09-05,hello,3,4" << std::endl;
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
          {"timestamp", "2022-09-06"},
          {"static_text", "hello"},
          {"static_categorical", "0"},
          {"sequential", "2"}};
}

SequentialClassifier getTrainedClassifier(const char* train_file_name) {
  auto classifier = makeSequentialClassifierForMockData();

  classifier.train(train_file_name, /* epochs= */ 5, /* learning_rate= */ 0.01);

  return classifier;
}

void assert_column_names(std::vector<std::string> column_names) {
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
    } else {
      ASSERT_EQ(
          std::count(column_names.begin(), column_names.end(), column_name), 1);
    }
  }
}

void assert_percentage_significance(
    std::vector<float> percentage_significances) {
  // assert the values are sorted in descending order of absolute values.
  bool isSorted = std::is_sorted(
      percentage_significances.begin(), percentage_significances.end(),
      [](float pair1, float pair2) { return abs(pair1) >= abs(pair2); });

  ASSERT_TRUE(isSorted);

  // assert the total sum of absolute values is close to 100.
  float total_percentage_sum = 0.0;

  for (auto percentage_significance : percentage_significances) {
    total_percentage_sum += std::abs(percentage_significance);
  }

  ASSERT_GT(total_percentage_sum, 0.99);
}

void assert_is_between(uint32_t value, uint32_t upper_limit) {
  ASSERT_GE(value, 0);
  ASSERT_LE(value, upper_limit);
}

void assert_indices_within_block(std::vector<std::string> column_names,
                                 std::vector<uint32_t> indices_within_block) {
  // the upper limit is caluclated based on the values mentioned in schema.
  // for the text block the default limit is 100000.
  for (uint32_t i = 0; i < indices_within_block.size(); i++) {
    if (column_names[i] == "timestamp") {
      assert_is_between(indices_within_block[i], 73);
    } else if (column_names[i] == "sequential") {
      assert_is_between(indices_within_block[i], 3);
    } else if (column_names[i] == "static_text") {
      assert_is_between(indices_within_block[i], 100000);
    } else {
      assert_is_between(indices_within_block[i], 5);
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

  auto [column_names, percentage_significances, indices_within_block] =
      classifier.explain(single_inference_input);

  // we will check how many times the column names are present in the vector.
  assert_column_names(column_names);

  // we will check the total percentage is close to 100 and the percentage
  // significance are sorted.
  assert_percentage_significance(percentage_significances);

  // we will check the indices within each block are within the limits.
  assert_indices_within_block(column_names, indices_within_block);

  std::remove(train_file_name);
}

}  // namespace thirdai::bolt::sequential_classifier::tests