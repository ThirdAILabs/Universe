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
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/Tabular.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <memory>
#include <stdexcept>

namespace thirdai::automl::tests {

// This is implemented as a macro instead of a function so that we can get line
// numbers when it fails. NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define ASSERT_TRANSFORM_TYPE(transform, type) \
  ASSERT_TRUE(std::dynamic_pointer_cast<type>(transform))

TEST(TabularTransformationTests, TextOnlyTransformation) {
  auto [transformation, outputs] = inputTransformations(
      /* data_types= */ {{"text", std::make_shared<data::TextDataType>()},
                         {"label",
                          std::make_shared<data::CategoricalDataType>()}},
      /* label_column= */ "label", /* temporal_relationships= */ {},
      /* options= */ data::TabularOptions(),
      /* should_update_history= */ false);

  ASSERT_TRANSFORM_TYPE(transformation, thirdai::data::TextTokenizer);
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs.at(0).indices(), "__featurized_input_indices__");
  // There should not be a specified values column, just indices.
  ASSERT_TRUE(outputs.at(0).values().has_value());
  ASSERT_EQ(outputs.at(0).values(), "__featurized_input_values__");
}

data::ColumnDataTypes getTabularDataTypes() {
  // We use a-f as the column names so that we know the order in which they will
  // be iterated over to create transformations since std::map is used.
  return {
      // Text
      {"a", std::make_shared<data::TextDataType>()},
      // Categorical
      {"b", std::make_shared<data::CategoricalDataType>()},
      // Multi categorical
      {"c", std::make_shared<data::CategoricalDataType>('-')},
      // Numerical
      {"d", std::make_shared<data::NumericalDataType>(0.0, 1.0)},
      // Sequence
      {"e", std::make_shared<data::SequenceDataType>()},
      // Date
      {"f", std::make_shared<data::DateDataType>()},
      // Extra for label
      {"label", std::make_shared<data::CategoricalDataType>()},
  };
}

thirdai::data::ColumnMap getInput() {
  return thirdai::data::ColumnMap::fromMapInput({
      {"a", "some text"},
      {"b", "cat_str"},
      {"c", "20-24-22"},
      {"d", "0.5"},
      {"e", "a b c d"},
      {"f", "2023-10-12"},
      {"label", "4"},
  });
}

void checkOutputs(const thirdai::data::OutputColumnsList& outputs) {
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs.at(0).indices(), "__featurized_input_indices__");
  ASSERT_EQ(outputs.at(0).values(), "__featurized_input_values__");
}

TEST(TabularTransformationTests, TabularTransformations) {
  auto [transformation, outputs] = inputTransformations(
      /* data_types= */ getTabularDataTypes(), /* label_column= */ "label",
      /* temporal_relationships= */ {}, /* options= */ data::TabularOptions(),
      /* should_update_history= */ false);

  checkOutputs(outputs);

  ASSERT_TRANSFORM_TYPE(transformation, thirdai::data::TransformationList);

  auto t_list = std::dynamic_pointer_cast<thirdai::data::TransformationList>(
                    transformation)
                    ->transformations();

  ASSERT_EQ(t_list.size(), 7);

  ASSERT_TRANSFORM_TYPE(t_list[0], thirdai::data::TextTokenizer);
  ASSERT_TRANSFORM_TYPE(t_list[1], thirdai::data::TextTokenizer);
  ASSERT_TRANSFORM_TYPE(t_list[2], thirdai::data::StringHash);
  ASSERT_TRANSFORM_TYPE(t_list[3], thirdai::data::HashPositionTransform);
  ASSERT_TRANSFORM_TYPE(t_list[4], thirdai::data::Date);
  ASSERT_TRANSFORM_TYPE(t_list[5], thirdai::data::Tabular);
  ASSERT_TRANSFORM_TYPE(t_list[6], thirdai::data::FeatureHash);

  auto tabular = std::dynamic_pointer_cast<thirdai::data::Tabular>(t_list[5]);

  ASSERT_EQ(tabular->numericalColumns().size(), 1);
  ASSERT_EQ(tabular->numericalColumns().at(0).name, "d");
  ASSERT_EQ(tabular->categoricalColumns().size(), 1);
  ASSERT_EQ(tabular->categoricalColumns().at(0).name, "b");

  auto fh_cols =
      std::dynamic_pointer_cast<thirdai::data::FeatureHash>(t_list[6])
          ->inputColumns();

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
  thirdai::data::State state;
  thirdai::data::TransformationList pipeline(t_list);
  pipeline.apply(getInput(), state);
}

TEST(TabularTransformationTests, TabularTransformationsTemporal) {
  data::TabularOptions options;
  options.contextual_columns = true;

  data::TemporalRelationships relationships = {
      {"b",
       {data::TemporalConfig::categorical("c", 2),
        data::TemporalConfig::categorical("label", 4)}}};

  auto [transformation, outputs] = inputTransformations(
      /* data_types= */ getTabularDataTypes(), /* label_column= */ "label",
      /* temporal_relationships= */ relationships, /* options= */ options,
      /* should_update_history= */ false);

  checkOutputs(outputs);

  ASSERT_TRANSFORM_TYPE(transformation, thirdai::data::TransformationList);

  auto t_list = std::dynamic_pointer_cast<thirdai::data::TransformationList>(
                    transformation)
                    ->transformations();

  ASSERT_EQ(t_list.size(), 11);

  ASSERT_TRANSFORM_TYPE(t_list[0], thirdai::data::TextTokenizer);
  ASSERT_TRANSFORM_TYPE(t_list[1], thirdai::data::StringHash);
  ASSERT_TRANSFORM_TYPE(t_list[2], thirdai::data::HashPositionTransform);
  ASSERT_TRANSFORM_TYPE(t_list[3], thirdai::data::Date);
  ASSERT_TRANSFORM_TYPE(t_list[4], thirdai::data::Tabular);
  ASSERT_TRANSFORM_TYPE(t_list[5], thirdai::data::StringToTimestamp);
  ASSERT_TRANSFORM_TYPE(t_list[6], thirdai::data::StringHash);
  ASSERT_TRANSFORM_TYPE(t_list[7], thirdai::data::CategoricalTemporal);
  ASSERT_TRANSFORM_TYPE(t_list[8], thirdai::data::StringHash);
  ASSERT_TRANSFORM_TYPE(t_list[9], thirdai::data::CategoricalTemporal);
  ASSERT_TRANSFORM_TYPE(t_list[10], thirdai::data::FeatureHash);

  auto tabular = std::dynamic_pointer_cast<thirdai::data::Tabular>(t_list[4]);

  ASSERT_EQ(tabular->numericalColumns().size(), 1);
  ASSERT_EQ(tabular->numericalColumns().at(0).name, "d");
  ASSERT_EQ(tabular->categoricalColumns().size(), 1);
  ASSERT_EQ(tabular->categoricalColumns().at(0).name, "b");

  auto fh_cols =
      std::dynamic_pointer_cast<thirdai::data::FeatureHash>(t_list[10])
          ->inputColumns();

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
  thirdai::data::State state;
  thirdai::data::TransformationList pipeline(t_list);
  pipeline.apply(getInput(), state);
}

TEST(TabularTransformationTests, CheckRejectsReservedColumns) {
  ASSERT_THROW(  // NOLINT clang-tidy doens't like ASSERT_THROW
      inputTransformations(
          /* data_types= */ {{"__text__",
                              std::make_shared<data::TextDataType>()}},
          /* label_column= */ "label", /* temporal_relationships= */ {},
          /* options= */ data::TabularOptions(),
          /* should_update_history= */ false),
      std::invalid_argument);
}

}  // namespace thirdai::automl::tests