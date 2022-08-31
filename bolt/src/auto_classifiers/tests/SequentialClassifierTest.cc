#include <bolt/src/auto_classifiers/sequential_classifier/SequentialClassifier.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialUtils.h>
#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <utility>
#include <vector>

namespace thirdai::bolt::sequential_classifier::tests {

void writeMockSequentialDataToFile(const std::string& train_file_name,
                                   const std::string& test_file_name) {
  std::string header = "user,target,timestamp,static_text,static_categorical";

  std::ofstream train_file(train_file_name);
  train_file << header << std::endl;
  train_file << "0,0,2022-08-29,hello,0" << std::endl;
  train_file << "0,1,2022-08-30,hello,1" << std::endl;
  train_file << "0,0,2022-08-31,hello,2" << std::endl;
  train_file << "0,1,2022-09-01,hello,3" << std::endl;

  std::ofstream test_file(test_file_name);
  test_file << header << std::endl;
  test_file << "0,0,2022-09-02,hello,0" << std::endl;
  test_file << "0,1,2022-09-03,hello,1" << std::endl;
  test_file << "0,0,2022-09-04,hello,2" << std::endl;
  test_file << "0,1,2022-09-05,hello,3" << std::endl;
}

SequentialClassifier makeSequentialClassifierForMockData() {
  CategoricalPair user = {"user", 5};
  CategoricalPair target = {"target", 5};
  std::string timestamp = "timestamp";
  std::vector<std::string> static_text = {"static_text"};
  std::vector<CategoricalTuple> static_categorical = {
      CategoricalPair("static_categorical", 5)};
  std::vector<SequentialTuple> sequential = {SequentialTriplet("target", 5, 3)};
  return {user,        std::move(target),  timestamp,
          static_text, static_categorical, sequential};
}

TEST(SequentialClassifierTest, TestLoadSave) {
  const char* train_file_name = "seq_class_train.csv";
  const char* test_file_name = "seq_class_test.csv";
  const char* model_save_file_name = "seq_class_save";

  writeMockSequentialDataToFile(train_file_name, test_file_name);

  auto classifier = makeSequentialClassifierForMockData();

  classifier.train(train_file_name, /* epochs= */ 5, /* learning_rate= */ 0.01);

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

}  // namespace thirdai::bolt::sequential_classifier::tests