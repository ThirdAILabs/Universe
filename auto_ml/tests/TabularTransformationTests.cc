#include "gtest/gtest.h"
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <auto_ml/src/featurization/TabularTransformations.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/Date.h>
#include <data/src/transformations/EncodePosition.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/Tabular.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <memory>
#include <stdexcept>

namespace thirdai::automl::tests {

// This is implemented as a macro instead of a function so that we can get line
// numbers when it fails. NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define ASSERT_TRANSFORM_TYPE(transform, type) \
  ASSERT_TRUE(std::dynamic_pointer_cast<type>(transform))

TEST(TabularTransformationTests, TextOnlyTransformation) {
  auto [transformation, outputs] = inputTransformations(
      /* data_types= */ {{"text", std::make_shared<TextDataType>()},
                         {"label", std::make_shared<CategoricalDataType>()}},
      /* label_column= */ "label", /* temporal_relationships= */ {},
      /* options= */ TabularOptions(),
      /* should_update_history= */ false);

  ASSERT_TRANSFORM_TYPE(transformation, data::TextTokenizer);
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs.at(0).indices(), "__featurized_input_indices__");
  // There should not be a specified values column, just indices.
  ASSERT_TRUE(outputs.at(0).values().has_value());
  ASSERT_EQ(outputs.at(0).values(), "__featurized_input_values__");
}

ColumnDataTypes getTabularDataTypes() {
  // We use a-f as the column names so that we know the order in which they will
  // be iterated over to create transformations since std::map is used.
  return {
      // Text
      {"a", std::make_shared<TextDataType>()},
      // Categorical
      {"b", std::make_shared<CategoricalDataType>()},
      // Multi categorical
      {"c", std::make_shared<CategoricalDataType>(10, "int", '-')},
      // Numerical
      {"d", std::make_shared<NumericalDataType>(0.0, 1.0)},
      // Sequence
      {"e", std::make_shared<SequenceDataType>()},
      // Date
      {"f", std::make_shared<DateDataType>()},
      // Extra for label
      {"label", std::make_shared<CategoricalDataType>()},
  };
}

data::ColumnMap getInput() {
  return data::ColumnMap::fromMapInput({
      {"a", "some text"},
      {"b", "cat_str"},
      {"c", "20-24-22"},
      {"d", "0.5"},
      {"e", "a b c d"},
      {"f", "2023-10-12"},
      {"label", "4"},
  });
}

void checkOutputs(const data::OutputColumnsList& outputs) {
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs.at(0).indices(), "__featurized_input_indices__");
  ASSERT_EQ(outputs.at(0).values(), "__featurized_input_values__");
}

TEST(TabularTransformationTests, TabularTransformations) {
  auto [transformation, outputs] = inputTransformations(
      /* data_types= */ getTabularDataTypes(), /* label_column= */ "label",
      /* temporal_relationships= */ {}, /* options= */ TabularOptions(),
      /* should_update_history= */ false);

  checkOutputs(outputs);

  ASSERT_TRANSFORM_TYPE(transformation, data::Pipeline);

  auto t_list = std::dynamic_pointer_cast<data::Pipeline>(transformation)
                    ->transformations();

  ASSERT_EQ(t_list.size(), 7);

  ASSERT_TRANSFORM_TYPE(t_list[0], data::TextTokenizer);
  ASSERT_TRANSFORM_TYPE(t_list[1], data::TextTokenizer);
  ASSERT_TRANSFORM_TYPE(t_list[2], data::StringHash);
  ASSERT_TRANSFORM_TYPE(t_list[3], data::HashPositionTransform);
  ASSERT_TRANSFORM_TYPE(t_list[4], data::Date);
  ASSERT_TRANSFORM_TYPE(t_list[5], data::Tabular);
  ASSERT_TRANSFORM_TYPE(t_list[6], data::FeatureHash);

  auto tabular = std::dynamic_pointer_cast<data::Tabular>(t_list[5]);

  ASSERT_EQ(tabular->numericalColumns().size(), 1);
  ASSERT_EQ(tabular->numericalColumns().at(0).name, "d");
  ASSERT_EQ(tabular->categoricalColumns().size(), 1);
  ASSERT_EQ(tabular->categoricalColumns().at(0).name, "b");

  auto fh_cols =
      std::dynamic_pointer_cast<data::FeatureHash>(t_list[6])->inputColumns();

  // The transformations on columns b and d are at the end because after
  // processing all data types we check if cross column pairgrams is enabled and
  // then decide if these columns are feature hashed or passed through cross
  // column pairgrams.
  std::vector<std::string> expected_fh_cols = {
      "__a_tokenized__", "__c_categorical__",   "__e_sequence__",
      "__f_date__",      "__tabular_columns__",
  };
  ASSERT_EQ(fh_cols, expected_fh_cols);

  // Check that transformations can process data without errors.
  data::State state;
  data::Pipeline pipeline(t_list);
  pipeline.apply(getInput(), state);
}

TEST(TabularTransformationTests, TabularTransformationsTemporal) {
  TabularOptions options;
  options.contextual_columns = true;

  TemporalRelationships relationships = {
      {"b",
       {TemporalConfig::categorical("c", 2),
        TemporalConfig::categorical("label", 4)}}};

  auto [transformation, outputs] = inputTransformations(
      /* data_types= */ getTabularDataTypes(), /* label_column= */ "label",
      /* temporal_relationships= */ relationships, /* options= */ options,
      /* should_update_history= */ false);

  checkOutputs(outputs);

  ASSERT_TRANSFORM_TYPE(transformation, data::Pipeline);

  auto t_list = std::dynamic_pointer_cast<data::Pipeline>(transformation)
                    ->transformations();

  ASSERT_EQ(t_list.size(), 10);

  ASSERT_TRANSFORM_TYPE(t_list[0], data::TextTokenizer);
  ASSERT_TRANSFORM_TYPE(t_list[1], data::StringHash);
  ASSERT_TRANSFORM_TYPE(t_list[2], data::HashPositionTransform);
  ASSERT_TRANSFORM_TYPE(t_list[3], data::Date);
  ASSERT_TRANSFORM_TYPE(t_list[4], data::Tabular);
  ASSERT_TRANSFORM_TYPE(t_list[5], data::StringToTimestamp);
  ASSERT_TRANSFORM_TYPE(t_list[6], data::StringHash);
  ASSERT_TRANSFORM_TYPE(t_list[7], data::CategoricalTemporal);
  ASSERT_TRANSFORM_TYPE(t_list[8], data::CategoricalTemporal);
  ASSERT_TRANSFORM_TYPE(t_list[9], data::FeatureHash);

  auto tabular = std::dynamic_pointer_cast<data::Tabular>(t_list[4]);

  ASSERT_EQ(tabular->numericalColumns().size(), 1);
  ASSERT_EQ(tabular->numericalColumns().at(0).name, "d");
  ASSERT_EQ(tabular->categoricalColumns().size(), 1);
  ASSERT_EQ(tabular->categoricalColumns().at(0).name, "b");

  auto fh_cols =
      std::dynamic_pointer_cast<data::FeatureHash>(t_list[9])->inputColumns();

  // The transformations on columns b and d are at the end because after
  // processing all data types we check if cross column pairgrams is enabled and
  // then decide if these columns are feature hashed or passed through cross
  // column pairgrams.
  std::vector<std::string> expected_fh_cols = {
      "__a_tokenized__",
      "__e_sequence__",
      "__f_date__",
      "__tabular_columns__",
      "__categorical_temporal_0__",
      "__categorical_temporal_1__",
  };
  ASSERT_EQ(fh_cols, expected_fh_cols);

  // Check that transformations can process data without errors.
  data::State state;
  data::Pipeline pipeline(t_list);
  pipeline.apply(getInput(), state);
}

TEST(TabularTransformationTests, CheckRejectsReservedColumns) {
  ASSERT_THROW(  // NOLINT clang-tidy doens't like ASSERT_THROW
      inputTransformations(
          /* data_types= */ {{"__text__", std::make_shared<TextDataType>()}},
          /* label_column= */ "label", /* temporal_relationships= */ {},
          /* options= */ TabularOptions(),
          /* should_update_history= */ false),
      std::invalid_argument);
}

}  // namespace thirdai::automl::tests