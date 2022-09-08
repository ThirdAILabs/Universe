#include <bolt/src/auto_classifiers/sequential_classifier/SequentialClassifier.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialUtils.h>
#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <optional>
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

SequentialClassifier makeSequentialClassifierForMockData() {
  CategoricalPair user = {"user", 5};
  CategoricalPair target = {"target", 5};
  std::string timestamp = "timestamp";
  std::vector<std::string> static_text = {"static_text"};
  std::vector<CategoricalPair> static_categorical = {{"static_categorical", 5}};
  std::vector<SequentialTriplet> sequential = {{"target", 5, 3}};
  return {user, target, timestamp, static_text, static_categorical, sequential};
}

/**
 * @brief Tests that sequential classifier works when:
 * 1) Target and static categorical columns have multiple classes
 * 2) We pass in a multi_class_delimiter argument.
 */
TEST(SequentialClassifierTest, TestLoadSaveMultiClass) {
  writeRowsToFile(TRAIN_FILE_NAME,
                  {"user,target,timestamp,static_text,static_categorical",
                   "0,0,2022-08-29,hello,0", "0,1,2022-08-30,hello,1",
                   "0,0,2022-08-31,hello,2", "0,1,2022-09-01,hello,3"});

  /*
    The test set is curated so that the classifier
    successfully runs only if it correctly parses multi-class
    categorical columns. Notice that the target and
    static_categorical columns contain classes that have been
    seen in the train set, delimited by spaces. If the classifier
    fails to parse multi-class categorical columns, these columns
    would be treated as new unique classes, which then throws an
    error since it would have exceeded the expected number of
    unique classes.
  */
  writeRowsToFile(TEST_FILE_NAME,
                  {"user,target,timestamp,static_text,static_categorical",
                   "0,0 1,2022-09-02,hello,0 1", "0,1 0,2022-09-03,hello,1 2",
                   "0,0 1,2022-09-04,hello,2 3", "0,1 0,2022-09-05,hello,3 0"});

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

  SequentialClassifier model(
      /* user= */ {"user", 1},
      /* target= */ {"target", 1},
      /* timestamp= */ "timestamp",
      /* static_text= */ {},
      /* static_categorical= */ {},
      /* sequential= */ {},
      /* multi_class_delim= */ ' ');

  /*
    In the train file, user column of the second row
    should be parsed as a new unique string "0 0" as opposed to
    two previously seen "0" strings delimited by a space. Thus,
    we expect that the test should fail.
  */
  assertFailsTraining(model);
}

}  // namespace thirdai::bolt::sequential_classifier::tests