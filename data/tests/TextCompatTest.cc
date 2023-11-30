#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/TextCompat.h>
#include <dataset/src/blocks/text/Text.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <limits>
#include <memory>
#include <random>

namespace thirdai::data::tests {

dataset::TextTokenizerPtr getTokenizer() {
  return dataset::CharKGramTokenizer::make(/*k=*/4);
}

dataset::TextEncoderPtr getEncoder() {
  return dataset::NGramEncoder::make(/*n=*/2);
}

static const size_t DIM = 7000;

using IndicesAndValues = std::pair<std::vector<std::vector<uint32_t>>,
                                   std::vector<std::vector<float>>>;

IndicesAndValues applyTransformation(std::vector<std::string> text) {
  TextCompat transform("txt", "indices", "values", getTokenizer(), getEncoder(),
                       /*lowercase=*/true,
                       /*encoding_dim=*/std::numeric_limits<uint32_t>::max(),
                       /*feature_hash_dim=*/DIM);

  ColumnMap columns({{"txt", ValueColumn<std::string>::make(std::move(text))}});

  columns = transform.applyStateless(columns);

  return {
      std::dynamic_pointer_cast<ArrayColumn<uint32_t>>(
          columns.getColumn("indices"))
          ->data(),
      std::dynamic_pointer_cast<ArrayColumn<float>>(columns.getColumn("values"))
          ->data(),
  };
}

IndicesAndValues applyBlock(const std::vector<std::string>& text) {
  auto block = dataset::TextBlock::make(
      dataset::ColumnIdentifier("text"), getTokenizer(), getEncoder(),
      /*lowercase=*/true, std::numeric_limits<uint32_t>::max());

  dataset::TabularFeaturizer featurizer(
      {dataset::BlockList({block}, /*hash_range=*/DIM)});

  featurizer.processHeader("text");
  auto rows = featurizer.featurize(text);

  IndicesAndValues output;
  for (const auto& row : rows.at(0)) {
    output.first.emplace_back(row.active_neurons, row.active_neurons + row.len);
    output.second.emplace_back(row.activations, row.activations + row.len);
  }

  return output;
}

std::vector<std::string> randomText() {
  std::mt19937 rng(2048);
  std::uniform_int_distribution<> sentence_len_dist(5, 20);
  std::uniform_int_distribution<> word_dist(3, 7);
  std::uniform_int_distribution<> char_dist(0, 51);

  std::vector<std::string> sentences;
  for (size_t sent = 0; sent < 1000; sent++) {
    std::string sentence;
    int sent_len = sentence_len_dist(rng);
    for (int word = 0; word < sent_len; word++) {
      if (word > 0) {
        sentence.push_back(' ');
      }
      int word_len = word_dist(rng);
      for (int c = 0; c < word_len; c++) {
        int offset = char_dist(rng);
        char base = offset < 26 ? 'a' : 'A';
        sentence.push_back(base + (offset % 26));
      }
    }
    sentences.push_back(sentence);
  }

  return sentences;
}

TEST(TextCompatTest, OutputsMatch) {
  std::vector<std::string> text = randomText();

  auto [transform_indices, transform_values] = applyTransformation(text);
  auto [block_indices, block_values] = applyBlock(text);

  EXPECT_EQ(transform_indices, block_indices);
  EXPECT_EQ(transform_values, block_values);
}

}  // namespace thirdai::data::tests