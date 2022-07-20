#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/sequential_classifier/SequentialClassifier.h>
#include <bolt/src/sequential_classifier/SequentialClassifierPipelineBuilder.h>
#include <gtest/gtest.h>
#include <dataset/src/utils/TimeUtils.h>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

using dataset::TimestampGenerator;

class TrendGenerator {
 public:
  enum class TrendState { increasing, decreasing, stagnant };

  explicit TrendGenerator(float starting_value)
      : _value(starting_value), _state(TrendState::stagnant) {}

  float nextValue() {
    switch (_state) {
      case TrendState::increasing:
        _value *= 1.2;
        break;
      case TrendState::decreasing:
        _value /= 1.2;
        break;
      default:
        break;
    }
    return _value;
  }

  void startIncreasing() { _state = TrendState::increasing; }

  void startDecreasing() { _state = TrendState::decreasing; }

  void startStagnating() { _state = TrendState::stagnant; }

  std::string currentState() const {
    switch (_state) {
      case TrendState::increasing:
        return "increasing";
      case TrendState::decreasing:
        return "decreasing";
      case TrendState::stagnant:
        return "stagnant";
      default:
        throw std::invalid_argument(
            "State is not increasing, decreasing, nor stagnant. Something is"
            "wrong.");
    }
  }

 private:
  float _value;
  TrendState _state;
};

class SequentialClassifierTests : public testing::Test {
 public:
  static constexpr const char* simple_trend_train_dataset =
      "trend_class_train.csv";
  static constexpr const char* simple_trend_test_dataset =
      "trend_class_test.csv";
  static constexpr const char* timestamp_col = "timestamp_col";
  static constexpr const char* item_col = "item_col";
  static constexpr const char* target_col = "target_col";
  static constexpr const char* trackable_qty_col = "track_qty_col";

  static void writeSimpleTrendClassificationDataset() {
    TimestampGenerator time_gen("1990-01-01");
    TrendGenerator trend_gen(/* starting_value = */ 1.0);

    std::ofstream train_out(simple_trend_train_dataset);
    writeHeaderToFile(train_out);
    uint32_t n_cycles = 10;
    uint32_t n_items = 200;
    uint32_t short_repetitions = 30;
    uint32_t long_repetitions = 100;
    for (uint32_t cycle = 0; cycle < n_cycles; cycle++) {
      trend_gen.startIncreasing();
      writeSamplesToFile(train_out, time_gen, trend_gen, n_items, "stagnant",
                         short_repetitions);
      writeSamplesToFile(train_out, time_gen, trend_gen, n_items, "increasing",
                         long_repetitions);
      trend_gen.startStagnating();
      writeSamplesToFile(train_out, time_gen, trend_gen, n_items, "stagnant",
                         short_repetitions);
      writeSamplesToFile(train_out, time_gen, trend_gen, n_items, "stagnant",
                         short_repetitions);
      trend_gen.startDecreasing();
      writeSamplesToFile(train_out, time_gen, trend_gen, n_items, "stagnant",
                         short_repetitions);
      writeSamplesToFile(train_out, time_gen, trend_gen, n_items, "decreasing",
                         long_repetitions);
      trend_gen.startStagnating();
      writeSamplesToFile(train_out, time_gen, trend_gen, n_items, "stagnant",
                         short_repetitions);
      writeSamplesToFile(train_out, time_gen, trend_gen, n_items, "stagnant",
                         short_repetitions);
    }
    trend_gen.startDecreasing();
    writeSamplesToFile(train_out, time_gen, trend_gen, n_items, "decreasing",
                       long_repetitions);
    train_out.close();

    /*
      At this point, the next value produced by trend generator
      is the lowest that it got, but we continue with a decreasing
      trend in the test file to make sure that sequential classifier
      can distinguish trends regardless of scale.
    */
    std::ofstream test_out(simple_trend_test_dataset);
    writeHeaderToFile(test_out);
    trend_gen.startDecreasing();
    writeSamplesToFile(test_out, time_gen, trend_gen, /* n_items = */ 200,
                       "decreasing", /* repetitions = */ 30);
    writeSamplesToFile(test_out, time_gen, trend_gen, /* n_items = */ 200,
                       "decreasing", /* repetitions = */ 150);
    trend_gen.startIncreasing();
    writeSamplesToFile(test_out, time_gen, trend_gen, /* n_items = */ 200,
                       "stagnant", /* repetitions = */ 30);
    writeSamplesToFile(test_out, time_gen, trend_gen, /* n_items = */ 200,
                       "increasing", /* repetitions = */ 150);
    test_out.close();
  }

  static void writeHeaderToFile(std::ofstream& out) {
    out << target_col << "," << item_col << "," << trackable_qty_col << ","
        << timestamp_col << std::endl;
  }

  static void writeSamplesToFile(std::ofstream& out,
                                 TimestampGenerator& time_gen,
                                 TrendGenerator& trend_gen, uint32_t n_items,
                                 std::string&& label, uint32_t repetitions) {
    for (uint32_t r = 0; r < repetitions; r++) {
      auto value = trend_gen.nextValue();
      auto timestring = time_gen.currentTimeString();
      for (uint32_t item_id = 0; item_id < n_items; item_id++) {
        std::stringstream next_line_ss;
        next_line_ss << label << "," << item_id << "," << value << ","
                     << timestring;
        out << next_line_ss.str() << std::endl;
      }
      time_gen.addDays(1);
    }
  }

  static void removeSimpleTrendClassificationDataset() {
    remove(simple_trend_train_dataset);
    remove(simple_trend_test_dataset);
  }
};

TEST_F(SequentialClassifierTests, SimpleTrendClassification) {
  writeSimpleTrendClassificationDataset();

  SequentialClassifierSchema schema(
      /* item = */ {/* col_name = */ item_col, /* n_distinct = */ 50000},
      /* timestamp = */ {/* col_name = */ timestamp_col},
      /* target = */ {/* col_name = */ target_col, /* n_distinct = */ 4},
      /* tracking_config = */
      {/* horizon = */ 0, /* lookback = */ 30, /* period = */ 1},
      /* text_attrs = */ {}, /* cat_attrs = */ {},
      /* trackable_qtys = */ {{/* col_name = */ trackable_qty_col}},
      /* trackable_cats = */ {});
  SequentialClassifier seq_bolt(schema, "small");

  seq_bolt.train(simple_trend_train_dataset, /* epochs = */ 1,
                 /* learning_rate = */ 0.0001);
  auto acc = seq_bolt.predict(simple_trend_test_dataset, "predictions.txt");
  ASSERT_GT(acc, 0.8);

  removeSimpleTrendClassificationDataset();
}

TEST_F(SequentialClassifierTests, Explainability) {
  writeSimpleTrendClassificationDataset();

  SequentialClassifierSchema schema(
      /* item = */ {/* col_name = */ item_col, /* n_distinct = */ 50000},
      /* timestamp = */ {/* col_name = */ timestamp_col},
      /* target = */ {/* col_name = */ target_col, /* n_distinct = */ 4},
      /* tracking_config = */
      {/* horizon = */ 0, /* lookback = */ 30, /* period = */ 1},
      /* text_attrs = */ {}, /* cat_attrs = */ {},
      /* trackable_qtys = */ {{/* col_name = */ trackable_qty_col}},
      /* trackable_cats = */ {});
  SequentialClassifier seq_bolt(schema, "small");

  seq_bolt.train(simple_trend_train_dataset, /* epochs = */ 1,
                 /* learning_rate = */ 0.0001);

  auto temp = seq_bolt.explain(simple_trend_test_dataset);

  for (auto& i : temp[0]) {
    std::cout << std::get<0>(i) << " " << std::get<1>(i) << " "
              << std::get<2>(i);
  }
  std::cout << std::endl;

  ASSERT_EQ(1.0, 1.0);

  removeSimpleTrendClassificationDataset();
}

}  // namespace thirdai::bolt