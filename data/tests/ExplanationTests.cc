#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/CrossColumnPairgrams.h>
#include <data/src/transformations/Date.h>
#include <data/src/transformations/EncodePosition.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/StringIDLookup.h>
#include <data/src/transformations/Tabular.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <limits>
#include <memory>
#include <optional>

namespace thirdai::data::tests {

void compareAllExplanations(const ExplanationMap& explanations,
                            const std::string& column,
                            const std::set<std::string>& expected) {
  auto explanation_msgs = explanations.explanationsForColumn(column);
  std::set<std::string> explanation_msgs_set(explanation_msgs.begin(),
                                             explanation_msgs.end());

  ASSERT_EQ(explanation_msgs_set, expected);
}

TEST(ExplanationTests, Binning) {
  BinningTransformation binning("aaa", "bbb", /* inclusive_min_value= */ 10,
                                /* exclusive_max_value= */ 20,
                                /* num_bins= */ 10);

  ColumnMap columns({{"aaa", ValueColumn<float>::make({14.7})}});

  State state;
  auto explanations = binning.explain(columns, state);

  ASSERT_EQ(explanations.explain("bbb", 4), "decimal 14.7 from column 'aaa'");
}

TEST(ExplanationTests, CrossColumnPairgrams) {
  CrossColumnPairgrams cross_column_pairgrams({"aa", "bb", "cc"}, "dd", 100000);

  ColumnMap columns({
      {"aa", ValueColumn<uint32_t>::make({0}, std::nullopt)},
      {"bb", ValueColumn<uint32_t>::make({10}, std::nullopt)},
      {"cc", ValueColumn<uint32_t>::make({100}, std::nullopt)},
  });

  State state;
  auto explanations = cross_column_pairgrams.explain(columns, state);

  std::set<std::string> expected_msgs = {
      "token 0 from column 'aa'",
      "token 0 from column 'aa' and token 10 from column 'bb'",
      "token 10 from column 'bb'",
      "token 0 from column 'aa' and token 100 from column 'cc'",
      "token 10 from column 'bb' and token 100 from column 'cc'",
      "token 100 from column 'cc'",
  };

  compareAllExplanations(explanations, "dd", expected_msgs);
}

TEST(ExplanationTests, Date) {
  Date date("aaa", "bbb");

  ColumnMap columns({{"aaa", ValueColumn<std::string>::make({"2023-10-12"})}});

  State state;
  auto explanations = date.explain(columns, state);

  size_t feature_offset = 0;
  ASSERT_EQ(explanations.explain("bbb", feature_offset + 0),
            "day of the week = 0 from column 'aaa'");
  feature_offset += 7;  // Days in week.

  ASSERT_EQ(explanations.explain("bbb", feature_offset + 9),
            "month of the year = 9 from column 'aaa'");
  feature_offset += 12;  // Months in year.

  ASSERT_EQ(explanations.explain("bbb", feature_offset + 1),
            "week of the month = 1 from column 'aaa'");
  feature_offset += 5;  // Max weeks in month.

  ASSERT_EQ(explanations.explain("bbb", feature_offset + 40),
            "week of the year = 40 from column 'aaa'");
}

TEST(ExplanationTests, HashedPosition) {
  HashPositionTransform encoder(
      "aaa", "bbb", /* hash_range= */ std::numeric_limits<uint32_t>::max());

  ColumnMap columns({{"aaa", ArrayColumn<uint32_t>::make({{1, 2, 3, 4}})}});

  State state;
  auto explanations = encoder.explain(columns, state);

  std::set<std::string> expected_msgs = {
      "token 1 from column 'aaa' at position 0",
      "token 2 from column 'aaa' at position 1",
      "token 3 from column 'aaa' at position 2",
      "token 4 from column 'aaa' at position 3",
  };

  compareAllExplanations(explanations, "bbb", expected_msgs);
}

TEST(ExplanationTests, OffsetPosition) {
  OffsetPositionTransform encoder("aaa", "bbb", /* max_num_tokens= */ 10);

  ColumnMap columns(
      {{"aaa", ArrayColumn<uint32_t>::make({{1, 2, 3, 4}}, /* dim= */ 10)}});

  State state;
  auto explanations = encoder.explain(columns, state);

  ASSERT_EQ(explanations.explain("bbb", 1),
            "token 1 from column 'aaa' at position 0");
  ASSERT_EQ(explanations.explain("bbb", 12),
            "token 2 from column 'aaa' at position 1");
  ASSERT_EQ(explanations.explain("bbb", 23),
            "token 3 from column 'aaa' at position 2");
  ASSERT_EQ(explanations.explain("bbb", 34),
            "token 4 from column 'aaa' at position 3");
}

TEST(ExplanationTests, FeatureHash) {
  FeatureHash feature_hash({"aaa", "bbb"}, "indices", "values",
                           /* hash_range= */ 100000);

  ColumnMap columns({{"aaa", ArrayColumn<uint32_t>::make({{1, 2, 3, 4}})},
                     {"bbb", ArrayColumn<float>::make({{10.1, 11.2, 12.3}})}});

  State state;
  auto explanations = feature_hash.explain(columns, state);

  std::set<std::string> expected_msgs = {
      "token 1 from column 'aaa'",      "token 2 from column 'aaa'",
      "token 3 from column 'aaa'",      "token 4 from column 'aaa'",
      "decimal 10.1 from column 'bbb'", "decimal 11.2 from column 'bbb'",
      "decimal 12.3 from column 'bbb'"};

  compareAllExplanations(explanations, "indices", expected_msgs);
}

TEST(ExplanationTests, StringCast) {
  TransformationList casts({
      std::make_shared<StringToToken>("a", "aa", /* dim= */ std::nullopt),
      std::make_shared<StringToDecimal>("b", "bb"),
      std::make_shared<StringToTimestamp>("c", "cc", /* format= */ "%Y-%m-%d"),
      std::make_shared<StringToTokenArray>("d", "dd", /* delimiter= */ ',',
                                           /* dim= */ std::nullopt),
      std::make_shared<StringToDecimalArray>("e", "ee", /* delimiter= */ ',',
                                             /* dim= */ std::nullopt),
  });

  ColumnMap columns({
      {"a", ValueColumn<std::string>::make({"64"})},
      {"b", ValueColumn<std::string>::make({"6.4"})},
      {"c", ValueColumn<std::string>::make({"2023-08-07"})},
      {"d", ValueColumn<std::string>::make({"6,4,2"})},
      {"e", ValueColumn<std::string>::make({"6.4,4.2,2.0"})},
  });

  State state;
  auto explanations = casts.explain(columns, state);

  ASSERT_EQ(explanations.explain("aa", 64), "token 64 from column 'a'");
  ASSERT_EQ(explanations.explain("bb", 0), "decimal 6.4 from column 'b'");
  ASSERT_EQ(explanations.explain("cc", 0),
            "timestamp 2023-08-07 from column 'c'");

  ASSERT_EQ(explanations.explain("dd", 6), "token 6 from column 'd'");
  ASSERT_EQ(explanations.explain("dd", 4), "token 4 from column 'd'");
  ASSERT_EQ(explanations.explain("dd", 2), "token 2 from column 'd'");

  ASSERT_EQ(explanations.explain("ee", 0), "decimal 6.4 from column 'e'");
  ASSERT_EQ(explanations.explain("ee", 1), "decimal 4.2 from column 'e'");
  ASSERT_EQ(explanations.explain("ee", 2), "decimal 2.0 from column 'e'");
}

TEST(ExplanationTests, StringHash) {
  StringHash string_hash("aaa", "bbb");

  ColumnMap columns({{"aaa", ValueColumn<std::string>::make({"some str"})}});

  State state;
  auto explanations = string_hash.explain(columns, state);

  auto output = string_hash.apply(columns, state);

  uint32_t feature = output.getValueColumn<uint32_t>("bbb")->value(0);

  ASSERT_EQ(explanations.explain("bbb", feature),
            "item 'some str' from column 'aaa'");
}

TEST(ExplanationTests, StringIDLookup) {
  StringIDLookup string_lookup("aaa", "bbb", "vocab",
                               /* max_vocab_size= */ std::nullopt,
                               /* delimiter= */ ';');

  ColumnMap columns({{"aaa", ValueColumn<std::string>::make({"x;y;z"})}});

  State state;
  string_lookup.apply(columns, state);  // To create the IDs.
  auto explanations = string_lookup.explain(columns, state);

  ASSERT_EQ(explanations.explain("bbb", 0), "item 'x' from column 'aaa'");
  ASSERT_EQ(explanations.explain("bbb", 1), "item 'y' from column 'aaa'");
  ASSERT_EQ(explanations.explain("bbb", 2), "item 'z' from column 'aaa'");
}

void testTabularExplanations(bool pairgrams,
                             const std::set<std::string>& expected) {
  Tabular tabular({NumericalColumn("bb", 0, 2, 2)},
                  {CategoricalColumn("aa"), CategoricalColumn("cc")}, "dd",
                  /* cross_column_pairgrams= */ pairgrams);

  ColumnMap columns({
      {"aa", ValueColumn<std::string>::make({"apple"})},
      {"bb", ValueColumn<std::string>::make({"1.7"})},
      {"cc", ValueColumn<std::string>::make({"88"})},
  });

  State state;
  auto explanations = tabular.explain(columns, state);

  compareAllExplanations(explanations, "dd", expected);
}

TEST(ExplanationTests, TabularWithPairgrams) {
  std::set<std::string> expected_msgs = {
      "category 'apple' from column 'aa'",
      "decimal 1.7 from column 'bb' and category 'apple' from column 'aa'",
      "decimal 1.7 from column 'bb'",
      "category 'apple' from column 'aa' and category '88' from column 'cc'",
      "decimal 1.7 from column 'bb' and category '88' from column 'cc'",
      "category '88' from column 'cc'",
  };

  testTabularExplanations(/* pairgrams= */ true, expected_msgs);
}

TEST(ExplanationTests, TabularWithoutPairgrams) {
  std::set<std::string> expected_msgs = {
      "category 'apple' from column 'aa'",
      "decimal 1.7 from column 'bb'",
      "category '88' from column 'cc'",
  };

  testTabularExplanations(/* pairgrams= */ false, expected_msgs);
}

TEST(ExplanationTests, TextTokenizer) {
  TextTokenizer tokenizer("a", "b", std::nullopt,
                          dataset::NaiveSplitTokenizer::make(),
                          dataset::NGramEncoder::make(1));

  ColumnMap columns({{"a", ValueColumn<std::string>::make({"the tree grew"})}});

  State state;
  auto explanations = tokenizer.explain(columns, state);

  std::set<std::string> expected_msgs = {"word 'the' from column 'a'",
                                         "word 'tree' from column 'a'",
                                         "word 'grew' from column 'a'"};

  compareAllExplanations(explanations, "b", expected_msgs);
}

TEST(ExplanationTests, ComposedTransformations) {
  TransformationList transformations({
      std::make_shared<TextTokenizer>("a", "words", std::nullopt,
                                      dataset::NaiveSplitTokenizer::make(),
                                      dataset::NGramEncoder::make(1)),
      std::make_shared<StringToDecimal>("b", "b_cast"),
      std::make_shared<BinningTransformation>("b_cast", "b_binned",
                                              /* inclusive_min_value= */ 10,
                                              /* exlusive_max_value= */ 20,
                                              /* num_bins= */ 10),
      std::make_shared<BinningTransformation>("c", "c_binned",
                                              /* inclusive_min_value= */ 5,
                                              /* exlusive_max_value= */ 10,
                                              /* num_bins= */ 5),
      std::make_shared<StringHash>("d", "hash"),
      std::make_shared<StringToTokenArray>("e", "tokens", /* delimiter= */ ',',
                                           /* dim= */ std::nullopt),
      std::make_shared<CrossColumnPairgrams>(
          std::vector<std::string>{"c_binned", "hash"}, "column_pairgrams",
          100000),
      std::make_shared<FeatureHash>(
          std::vector<std::string>{"words", "b_binned", "column_pairgrams",
                                   "tokens"},
          "indices", "values", /* hash_range= */ 100000),
  });

  ColumnMap columns({
      {"a", ValueColumn<std::string>::make({"we baked cookies"})},
      {"b", ValueColumn<std::string>::make({"16.4"})},
      {"c", ValueColumn<float>::make({7.5})},
      {"d", ValueColumn<std::string>::make({"to_hash"})},
      {"e", ValueColumn<std::string>::make({"6,4,2"})},
  });

  State state;
  auto explanations = transformations.explain(columns, state);

  std::set<std::string> expected_msgs = {
      "word 'we' from column 'a'",
      "word 'baked' from column 'a'",
      "word 'cookies' from column 'a'",
      "decimal 16.4 from column 'b'",
      "token 6 from column 'e'",
      "token 4 from column 'e'",
      "token 2 from column 'e'",
      "decimal 7.5 from column 'c'",
      "item 'to_hash' from column 'd'",
      "decimal 7.5 from column 'c' and item 'to_hash' from column 'd'",
  };

  compareAllExplanations(explanations, "indices", expected_msgs);
}

}  // namespace thirdai::data::tests