#include <bolt/src/auto_classifiers/sequential_classifier/SequentialClassifier.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialUtils.h>
#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
<<<<<<< HEAD
#include <unordered_map>
=======
#include <optional>
>>>>>>> 56f2b447317f6447c102498eb69c1187140b7e50
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

void assertSuccessfulLoadSave(SequentialClassifier& model) {
  model.train(TRAIN_FILE_NAME, /* epochs= */ 5, /* learning_rate= */ 0.01);

  // Save before making original prediction so both calls to predict() use the
  // same starting data states (vocabulary and item history).
  model.save(MODEL_SAVE_FILE_NAME);
  auto loaded_model = SequentialClassifier::load(MODEL_SAVE_FILE_NAME);

  auto original_model_results =
      model.predict(TEST_FILE_NAME, /* metrics= */ {"recall@1"});

  auto loaded_model_results =
      loaded_model->predict(TEST_FILE_NAME, /* metrics= */ {"recall@1"});

  ASSERT_EQ(original_model_results["recall@1"],
            loaded_model_results["recall@1"]);

  std::remove(TRAIN_FILE_NAME);
  std::remove(TEST_FILE_NAME);
  std::remove(MODEL_SAVE_FILE_NAME);
}

void assertFailsTraining(SequentialClassifier& model) {
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      model.train(TRAIN_FILE_NAME, /* epochs= */ 5, /* learning_rate= */ 0.01),
      std::invalid_argument);

  std::remove(TRAIN_FILE_NAME);
}

std::unordered_map<std::string, std::string>
mockSequentialSampleForPredictSingle() {
  return {{"user", "0"},
          {"target", "0"},
          {"timestamp", "2022-09-06"},
          {"static_text", "hello"},
          {"static_categorical", "0"}};
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

/**
 * @brief Tests that sequential classifier works when:
 * 1) Target and static categorical columns have multiple classes
 * 2) We pass in a multi_class_delimiter argument.
 */
TEST(SequentialClassifierTest, TestLoadSaveMultiClass) {
  /*
    The train set is curated so that the classifier
    successfully runs only if it correctly parses multi-class
    categorical columns. Notice that in the last two samples,
    the target and static_categorical columns contain classes that
    have been seen in the previous samples, delimited by spaces.
    If the classifier fails to parse multi-class categorical columns,
    these columns would be treated as new unique classes, which then
    throws an error since it would have exceeded the expected number of
    unique classes.
  */
  writeRowsToFile(TRAIN_FILE_NAME,
                  {"user,target,timestamp,static_text,static_categorical",
                   "0,0,2022-08-29,hello,0", "0,1,2022-08-30,hello,1",
                   "0,0,2022-08-31,hello,2", "0,1,2022-09-01,hello,3",
                   "0,0 1,2022-09-02,hello,0 1", "0,1 0,2022-09-03,hello,1 2"});

<<<<<<< HEAD
  writeMockSequentialDataToFile(train_file_name, test_file_name);
  auto predict_single_sample = mockSequentialSampleForPredictSingle();
=======
  writeRowsToFile(TEST_FILE_NAME,
                  {"user,target,timestamp,static_text,static_categorical",
                   "0,0 1,2022-09-04,hello,2 3", "0,1 0,2022-09-05,hello,3 0"});
>>>>>>> 56f2b447317f6447c102498eb69c1187140b7e50

  SequentialClassifier model(
      /* user= */ {"user", 1},
      /* target= */ {"target", 2},
      /* timestamp= */ "timestamp",
      /* static_text= */ {"static_text"},
      /* static_categorical= */ {{"static_categorical", 4}},
      /* sequential= */ {{"target", 2, 3}},
      /* multi_class_delim= */ ' ');

  assertSuccessfulLoadSave(model);
}

/**
 * @brief Tests that sequential classifier works properly when:
 * 1) Each column in the dataset only has a single value
 * 2) We don't pass the optional multi_class_delim argument
 */
TEST(SequentialClassifierTest, TestLoadSaveNoMultiClassDelim) {
  writeRowsToFile(TRAIN_FILE_NAME,
                  {"user,target,timestamp,static_text,static_categorical",
                   "0,0,2022-08-29,hello,0", "0,1,2022-08-30,hello,1",
                   "0,0,2022-08-31,hello,2", "0,1,2022-09-01,hello,3"});

<<<<<<< HEAD
  auto original_model_results =
      classifier.predict(test_file_name, /* metrics= */ {"recall@1"});
  auto original_model_single_output =
      classifier.predictSingle(predict_single_sample);

  auto loaded_model_results =
      new_classifier->predict(test_file_name, /* metrics= */ {"recall@1"});
  auto loaded_model_single_output =
      new_classifier->predictSingle(predict_single_sample);
=======
  writeRowsToFile(TEST_FILE_NAME,
                  {"user,target,timestamp,static_text,static_categorical",
                   "0,0,2022-09-02,hello,0", "0,1,2022-09-03,hello,1",
                   "0,0,2022-09-04,hello,2", "0,1,2022-09-05,hello,3"});

  SequentialClassifier model(
      /* user= */ {"user", 1},
      /* target= */ {"target", 2},
      /* timestamp= */ "timestamp",
      /* static_text= */ {"static_text"},
      /* static_categorical= */ {{"static_categorical", 4}},
      /* sequential= */ {{"target", 2, 3}});

  assertSuccessfulLoadSave(model);
}

/**
 * @brief Tests that sequential classifier does not parse static categorical
 * columns into multiple classes if we don't pass in a multi_class_delim
 * argument.
 */
TEST(SequentialClassifierTest, TestNoMultiClassCategoricalIfNoDelimiter) {
  writeRowsToFile(TRAIN_FILE_NAME,
                  {
                      "user,target,timestamp,static_categorical",
                      "0,0,2022-08-29,0",
                      "0,1,2022-08-30,0 0",
                  });

  SequentialClassifier model(
      /* user= */ {"user", 1},
      /* target= */ {"target", 2},
      /* timestamp= */ "timestamp",
      /* static_text= */ {},
      /* static_categorical= */ {{"static_categorical", 1}},
      /* sequential= */ {{"target", 2, 3}});  // We do not pass the optional
                                              // multi_class_delim argument

  /*
    In the train file, static_categorical column of the second row
    should be parsed as a new unique string "0 0" as opposed to
    two previously seen "0" strings delimited by a space. Thus,
    we expect that the test should fail.
  */
  assertFailsTraining(model);
}

/**
 * @brief Tests that sequential classifier does not parse sequential
 * columns into multiple classes if we don't pass in a multi_class_delim
 * argument.
 */
TEST(SequentialClassifierTest, TestNoMultiClassSequentialIfNoDelimiter) {
  writeRowsToFile(TRAIN_FILE_NAME, {
                                       "user,target,timestamp,sequential",
                                       "0,0,2022-08-29,0",
                                       "0,1,2022-08-30,0 0",
                                   });

  SequentialClassifier model(
      /* user= */ {"user", 1},
      /* target= */ {"target", 2},
      /* timestamp= */ "timestamp",
      /* static_text= */ {},
      /* static_categorical= */ {},
      /* sequential= */ {{"sequential", 1, 3}});  // We do not pass the optional
                                                  // multi_class_delim argument

  /*
    In the train file, sequential column of the second row
    should be parsed as a new unique string "0 0" as opposed to
    two previously seen "0" strings delimited by a space. Thus,
    we expect that the test should fail.
  */
  assertFailsTraining(model);
}

/**
 * @brief Tests that sequential classifier does not parse target
 * columns into multiple classes if we don't pass in a multi_class_delim
 * argument.
 */
TEST(SequentialClassifierTest, TestNoMultiClassTargetIfNoDelimiter) {
  writeRowsToFile(TRAIN_FILE_NAME, {
                                       "user,target,timestamp",
                                       "0,0,2022-08-29",
                                       "0,0 0,2022-08-30",
                                   });

  SequentialClassifier model(
      /* user= */ {"user", 1},
      /* target= */ {"target", 1},
      /* timestamp= */ "timestamp");  // We do not pass the optional
                                      // multi_class_delim argument

  /*
    In the train file, target column of the second row
    should be parsed as a new unique string "0 0" as opposed to
    two previously seen "0" strings delimited by a space. Thus,
    we expect that the test should fail.
  */
  assertFailsTraining(model);
}

/**
 * @brief Tests that sequential classifier does not parse user
 * columns into multiple classes even if we pass in a multi_class_delim
 * argument.
 */
TEST(SequentialClassifierTest, TestNeverMultiClassUser) {
  writeRowsToFile(TRAIN_FILE_NAME, {
                                       "user,target,timestamp",
                                       "0,0,2022-08-29",
                                       "0 0,0,2022-08-30",
                                   });
>>>>>>> 56f2b447317f6447c102498eb69c1187140b7e50

  SequentialClassifier model(
      /* user= */ {"user", 1},
      /* target= */ {"target", 1},
      /* timestamp= */ "timestamp",
      /* static_text= */ {},
      /* static_categorical= */ {},
      /* sequential= */ {},
      /* multi_class_delim= */ ' ');

<<<<<<< HEAD
  ASSERT_EQ(original_model_single_output.isDense(), true);
  ASSERT_EQ(loaded_model_single_output.isDense(), true);
  ASSERT_EQ(original_model_single_output.len, loaded_model_single_output.len);

  for (uint32_t pos = 0; pos < original_model_single_output.len; pos++) {
    ASSERT_EQ(original_model_single_output.activations[pos],
              loaded_model_single_output.activations[pos]);
  }

  std::remove(train_file_name);
  std::remove(test_file_name);
  std::remove(model_save_file_name);
=======
  /*
    In the train file, user column of the second row
    should be parsed as a new unique string "0 0" as opposed to
    two previously seen "0" strings delimited by a space. Thus,
    we expect that the test should fail.
  */
  assertFailsTraining(model);
>>>>>>> 56f2b447317f6447c102498eb69c1187140b7e50
}

}  // namespace thirdai::bolt::sequential_classifier::tests