#include "gtest/gtest.h"
#include <auto_ml/src/featurization/TabularTransformations.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/CrossColumnPairgrams.h>
#include <data/src/transformations/Date.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/Sequence.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <memory>

namespace thirdai::automl::tests {

// This is implemented as a macro instead of a function so that we can get line
// numbers when it fails (clang-tidy doesn't like macros) NOLINTNEXTLINE
#define ASSERT_TRANSFORM_TYPE(transform, type) \
  ASSERT_TRUE(std::dynamic_pointer_cast<type>(transform))

TEST(TabularTransformationTests, TextOnlyTransformation) {
  auto [transformation, outputs] = inputTransformations(
      {{"text", std::make_shared<data::TextDataType>()},
       {"label", std::make_shared<data::CategoricalDataType>()}},
      "label", {}, data::TabularOptions(), false);

  ASSERT_TRANSFORM_TYPE(transformation, thirdai::data::TextTokenizer);
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs.at(0).first, "__text_tokenized__");
  ASSERT_FALSE(outputs.at(0).second.has_value());
}

data::ColumnDataTypes getTabularDataTypes() {
  std::pair<double, double> range = {0.0, 1.0};
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
      {"d", std::make_shared<data::NumericalDataType>(range)},
      // Sequence
      {"e", std::make_shared<data::SequenceDataType>()},
      // Date
      {"f", std::make_shared<data::DateDataType>()},
      // Extra for label
      {"label", std::make_shared<data::CategoricalDataType>()},
  };
}

void checkOutputs(const thirdai::data::IndexValueColumnList& outputs) {
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs.at(0).first, "__featurized_input_indices__");
  ASSERT_EQ(outputs.at(0).second, "__featurized_input_values__");
}

TEST(TabularTransformationTests, TabularTransformations) {
  auto [transformation, outputs] = inputTransformations(
      getTabularDataTypes(), "label", {}, data::TabularOptions(), false);

  checkOutputs(outputs);

  ASSERT_TRANSFORM_TYPE(transformation, thirdai::data::TransformationList);

  auto t_list = std::dynamic_pointer_cast<thirdai::data::TransformationList>(
                    transformation)
                    ->transformations();

  ASSERT_EQ(t_list.size(), 7);

  ASSERT_TRANSFORM_TYPE(t_list[0], thirdai::data::TextTokenizer);
  ASSERT_TRANSFORM_TYPE(t_list[1], thirdai::data::StringHash);
  ASSERT_TRANSFORM_TYPE(t_list[2], thirdai::data::TextTokenizer);
  ASSERT_TRANSFORM_TYPE(t_list[3], thirdai::data::BinningTransformation);
  ASSERT_TRANSFORM_TYPE(t_list[4], thirdai::data::Sequence);
  ASSERT_TRANSFORM_TYPE(t_list[5], thirdai::data::Date);
  ASSERT_TRANSFORM_TYPE(t_list[6], thirdai::data::FeatureHash);

  auto fh_cols =
      std::dynamic_pointer_cast<thirdai::data::FeatureHash>(t_list[6])
          ->inputColumns();

  // The transformations on columns b and d are at the end because after
  // processing all data types we check if cross column pairgrams is enabled and
  // then decide if these columns are feature hashed or passed through cross
  // column pairgrams.
  std::vector<std::string> expected_fh_cols = {
      "__a_tokenized__", "__c_categorical__", "__e_sequence__",
      "__f_date__", "__b_categorical__", "__d_binned__",
  };
  ASSERT_EQ(fh_cols, expected_fh_cols);
}

TEST(TabularTransformationTests, TabularTransformationsCrossColumnPairgrams) {
  data::TabularOptions options;
  options.contextual_columns = true;
  auto [transformation, outputs] =
      inputTransformations(getTabularDataTypes(), "label", {}, options, false);

  checkOutputs(outputs);

  ASSERT_TRANSFORM_TYPE(transformation, thirdai::data::TransformationList);

  auto t_list = std::dynamic_pointer_cast<thirdai::data::TransformationList>(
                    transformation)
                    ->transformations();

  ASSERT_EQ(t_list.size(), 8);

  ASSERT_TRANSFORM_TYPE(t_list[0], thirdai::data::TextTokenizer);
  ASSERT_TRANSFORM_TYPE(t_list[1], thirdai::data::StringHash);
  ASSERT_TRANSFORM_TYPE(t_list[2], thirdai::data::TextTokenizer);
  ASSERT_TRANSFORM_TYPE(t_list[3], thirdai::data::BinningTransformation);
  ASSERT_TRANSFORM_TYPE(t_list[4], thirdai::data::Sequence);
  ASSERT_TRANSFORM_TYPE(t_list[5], thirdai::data::Date);
  ASSERT_TRANSFORM_TYPE(t_list[6], thirdai::data::CrossColumnPairgrams);
  ASSERT_TRANSFORM_TYPE(t_list[7], thirdai::data::FeatureHash);

  auto ccp_cols =
      std::dynamic_pointer_cast<thirdai::data::CrossColumnPairgrams>(t_list[6])
          ->inputColumns();

  std::vector<std::string> expected_ccp_cols = {
      "__b_categorical__",
      "__d_binned__",
  };
  ASSERT_EQ(ccp_cols, expected_ccp_cols);

  auto fh_cols =
      std::dynamic_pointer_cast<thirdai::data::FeatureHash>(t_list[7])
          ->inputColumns();

  // The transformations on columns b and d are at the end because after
  // processing all data types we check if cross column pairgrams is enabled and
  // then decide if these columns are feature hashed or passed through cross
  // column pairgrams.
  std::vector<std::string> expected_fh_cols = {
      "__a_tokenized__", "__c_categorical__",      "__e_sequence__",
      "__f_date__", "__contextual_columns__",
  };
  ASSERT_EQ(fh_cols, expected_fh_cols);
}

TEST(TabularTransformationTests, TabularTransformationsTemporal) {
  data::TabularOptions options;
  options.contextual_columns = true;

  data::TemporalRelationships relationships = {
      {"b",
       {data::TemporalConfig::categorical("c", 2),
        data::TemporalConfig::categorical("label", 4)}}};

  auto [transformation, outputs] = inputTransformations(
      getTabularDataTypes(), "label", relationships, options, false);

  checkOutputs(outputs);

  ASSERT_TRANSFORM_TYPE(transformation, thirdai::data::TransformationList);

  auto t_list = std::dynamic_pointer_cast<thirdai::data::TransformationList>(
                    transformation)
                    ->transformations();

  ASSERT_EQ(t_list.size(), 10);

  ASSERT_TRANSFORM_TYPE(t_list[0], thirdai::data::TextTokenizer);
  ASSERT_TRANSFORM_TYPE(t_list[1], thirdai::data::StringHash);
  ASSERT_TRANSFORM_TYPE(t_list[2], thirdai::data::BinningTransformation);
  ASSERT_TRANSFORM_TYPE(t_list[3], thirdai::data::Sequence);
  ASSERT_TRANSFORM_TYPE(t_list[4], thirdai::data::Date);
  ASSERT_TRANSFORM_TYPE(t_list[5], thirdai::data::CrossColumnPairgrams);
  ASSERT_TRANSFORM_TYPE(t_list[6], thirdai::data::StringToTimestamp);
  ASSERT_TRANSFORM_TYPE(t_list[7], thirdai::data::CategoricalTemporal);
  ASSERT_TRANSFORM_TYPE(t_list[8], thirdai::data::CategoricalTemporal);
  ASSERT_TRANSFORM_TYPE(t_list[9], thirdai::data::FeatureHash);

  auto ccp_cols =
      std::dynamic_pointer_cast<thirdai::data::CrossColumnPairgrams>(t_list[5])
          ->inputColumns();

  std::vector<std::string> expected_ccp_cols = {
      "__b_categorical__",
      "__d_binned__",
  };
  ASSERT_EQ(ccp_cols, expected_ccp_cols);

  auto fh_cols =
      std::dynamic_pointer_cast<thirdai::data::FeatureHash>(t_list[9])
          ->inputColumns();

  // The transformations on columns b and d are at the end because after
  // processing all data types we check if cross column pairgrams is enabled and
  // then decide if these columns are feature hashed or passed through cross
  // column pairgrams.
  std::vector<std::string> expected_fh_cols = {
      "__a_tokenized__",
      "__e_sequence__",
      "__f_date__",
      "__contextual_columns__",
      "__categorical_temporal_0__",
      "__categorical_temporal_1__",
  };
  ASSERT_EQ(fh_cols, expected_fh_cols);
}

}  // namespace thirdai::automl::tests