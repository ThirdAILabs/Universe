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

using IndicesAndValues = std::pair<std::vector<std::vector<uint32_t>>,
                                   std::vector<std::vector<float>>>;

IndicesAndValues applyTransformation(const dataset::TextTokenizerPtr& tokenizer,
                                     const dataset::TextEncoderPtr& encoder,
                                     size_t encoding_dim, size_t hash_range,
                                     bool lowercase,
                                     std::vector<std::string> text) {
  TextCompat transform("txt", "indices", "values", tokenizer, encoder,
                       lowercase, encoding_dim, hash_range);

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

IndicesAndValues applyBlock(const dataset::TextTokenizerPtr& tokenizer,
                            const dataset::TextEncoderPtr& encoder,
                            size_t encoding_dim, size_t hash_range,
                            bool lowercase,
                            const std::vector<std::string>& text) {
  auto block =
      dataset::TextBlock::make(dataset::ColumnIdentifier("text"), tokenizer,
                               encoder, lowercase, encoding_dim);

  // For the featurize we are choosing a delimiter that we ensure is not in the
  // samples to featurize, that way we ensure that there is only one column.
  dataset::TabularFeaturizer featurizer(
      {dataset::BlockList({block}, /*hash_range=*/hash_range)},
      /*has_header=*/false, /*delimiter=*/'%');

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
  std::uniform_int_distribution<> word_len_dist(3, 7);

  std::vector<char> special_chars = {'?', '!', ',', '.', '"', '\'', '#',
                                     '&', '*', '(', ')', '-', ';',  ':'};
  std::uniform_int_distribution<> char_dist(0, 52 + special_chars.size() - 1);

  std::vector<std::string> sentences;
  for (size_t sent = 0; sent < 300; sent++) {
    std::string sentence;
    int sent_len = sentence_len_dist(rng);
    for (int word = 0; word < sent_len; word++) {
      if (word > 0) {
        sentence.push_back(' ');
      }
      int word_len = word_len_dist(rng);
      for (int c = 0; c < word_len; c++) {
        int offset = char_dist(rng);
        if (offset < 26) {
          sentence.push_back('a' + offset);
        } else if (offset < 52) {
          sentence.push_back('A' + offset - 26);
        } else {
          sentence.push_back(special_chars[offset - 52]);
        }
      }
    }
    sentences.push_back(sentence);
  }

  return sentences;
}

std::vector<dataset::TextTokenizerPtr> getTokenizers() {
  return {dataset::CharKGramTokenizer::make(/*k=*/4),
          dataset::WordPunctTokenizer::make(),
          dataset::NaiveSplitTokenizer::make(/*delimiter=*/' ')};
}

std::vector<dataset::TextEncoderPtr> getEncoders() {
  return {dataset::NGramEncoder::make(/*n=*/1),
          dataset::NGramEncoder::make(/*n=*/2),
          dataset::PairGramEncoder::make()};
}

std::vector<size_t> getEncodingDims() {
  return {100000, 1000000, std::numeric_limits<uint32_t>::max()};
}

std::vector<size_t> getHashRanges() { return {1000, 7000, 100000}; }

TEST(TextCompatTest, OutputsMatch) {
  std::vector<std::string> text = randomText();

  auto tokenizers = getTokenizers();
  auto encoders = getEncoders();
  auto encoding_dims = getEncodingDims();
  auto hash_ranges = getHashRanges();
  std::vector<bool> lowercases = {true, false};

  for (const auto& tokenizer : tokenizers) {
    for (const auto& encoder : encoders) {
      for (auto encoding_dim : encoding_dims) {
        for (auto hash_range : hash_ranges) {
          for (auto lowercase : lowercases) {
            auto [transform_indices, transform_values] = applyTransformation(
                tokenizer, encoder, encoding_dim, hash_range, lowercase, text);

            auto [block_indices, block_values] = applyBlock(
                tokenizer, encoder, encoding_dim, hash_range, lowercase, text);

            ASSERT_EQ(transform_indices, block_indices);
            ASSERT_EQ(transform_values, block_values);
          }
        }
      }
    }
  }
}

}  // namespace thirdai::data::tests