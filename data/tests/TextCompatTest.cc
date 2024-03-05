#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/TextCompat.h>
#include <data/tests/MockDataSource.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/text/Text.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <dataset/src/mach/MachBlock.h>
#include <dataset/src/mach/MachIndex.h>
#include <limits>
#include <memory>
#include <random>
#include <sstream>

namespace thirdai::data::tests {

using IndicesAndValues = std::pair<std::vector<std::vector<uint32_t>>,
                                   std::vector<std::vector<float>>>;

IndicesAndValues applyTransformation(const dataset::TextTokenizerPtr& tokenizer,
                                     const dataset::TextEncoderPtr& encoder,
                                     size_t encoding_dim, size_t hash_range,
                                     bool lowercase,
                                     std::vector<std::string> text,
                                     bool serialize = false) {
  TextCompat transform("txt", "indices", "values", tokenizer, encoder,
                       lowercase, encoding_dim, hash_range);

  if (serialize) {
    std::stringstream buffer;
    ar::serialize(transform.toArchive(), buffer);
    transform = TextCompat(*ar::deserialize(buffer));
  }

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

std::vector<std::string> randomText(size_t n_lines) {
  std::mt19937 rng(2048);

  std::uniform_int_distribution<> sentence_len_dist(5, 20);
  std::uniform_int_distribution<> word_len_dist(3, 7);

  std::vector<char> special_chars = {'?', '!', ',', '.', '"', '\'', '#',
                                     '&', '*', '(', ')', '-', ';',  ':'};
  std::uniform_int_distribution<> char_dist(0, 52 + special_chars.size() - 1);

  std::vector<std::string> sentences;
  for (size_t sent = 0; sent < n_lines; sent++) {
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
  std::vector<std::string> text = randomText(/*n_lines=*/300);

  for (const auto& tokenizer : getTokenizers()) {
    for (const auto& encoder : getEncoders()) {
      for (auto encoding_dim : getEncodingDims()) {
        for (auto hash_range : getHashRanges()) {
          for (auto lowercase : {true, false}) {
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

IndicesAndValues tensorToIndicesAndValues(const bolt::TensorPtr& tensor) {
  IndicesAndValues output;
  for (size_t i = 0; i < tensor->batchSize(); i++) {
    const BoltVector& row = tensor->getVector(i);
    output.first.emplace_back(row.active_neurons, row.active_neurons + row.len);
    output.second.emplace_back(row.activations, row.activations + row.len);
  }
  return output;
}

std::tuple<IndicesAndValues, IndicesAndValues, IndicesAndValues>
applyTransformations(const dataset::DataSourcePtr& data,
                     const MachIndexPtr& mach_index,
                     const dataset::TextTokenizerPtr& tokenizer,
                     const dataset::TextEncoderPtr& encoder,
                     size_t encoding_dim, size_t hash_range, bool lowercase) {
  auto pipeline =
      Pipeline::make()
          ->then(std::make_shared<TextCompat>("TEXT", "indices", "values",
                                              tokenizer, encoder, lowercase,
                                              encoding_dim, hash_range))
          ->then(
              std::make_shared<StringToTokenArray>("IDS", "labels", ':', 100))
          ->then(std::make_shared<MachLabel>("labels", "mach_labels"));

  auto state = std::make_shared<State>(mach_index);

  Loader loader(
      CsvIterator::make(data, ','), pipeline, state,
      {OutputColumns::sparse("indices", "values")},
      {OutputColumns::sparse("labels"), OutputColumns::sparse("mach_labels")},
      1000,
      /*shuffle=*/false, /*verbose=*/false);

  auto [inputs, labels] = loader.all();

  EXPECT_EQ(inputs.size(), 1);
  EXPECT_EQ(labels.size(), 1);

  return {tensorToIndicesAndValues(inputs.at(0).at(0)),
          tensorToIndicesAndValues(labels.at(0).at(0)),
          tensorToIndicesAndValues(labels.at(0).at(1))};
}

IndicesAndValues batchToIndicesAndValues(const BoltBatch& batch) {
  IndicesAndValues output;
  for (const auto& row : batch) {
    output.first.emplace_back(row.active_neurons, row.active_neurons + row.len);
    output.second.emplace_back(row.activations, row.activations + row.len);
  }
  return output;
}

std::tuple<IndicesAndValues, IndicesAndValues, IndicesAndValues> applyBlocks(
    const dataset::DataSourcePtr& data, const MachIndexPtr& mach_index,
    const dataset::TextTokenizerPtr& tokenizer,
    const dataset::TextEncoderPtr& encoder, size_t encoding_dim,
    size_t hash_range, bool lowercase) {
  auto text =
      dataset::TextBlock::make(dataset::ColumnIdentifier("TEXT"), tokenizer,
                               encoder, lowercase, encoding_dim);

  auto labels = dataset::NumericalCategoricalBlock::make(
      dataset::ColumnIdentifier("IDS"), 100, ':');

  auto make_labels = dataset::mach::MachBlock::make(
      dataset::ColumnIdentifier("IDS"), mach_index, ':');

  auto featurizer = dataset::TabularFeaturizer::make(
      {dataset::BlockList({text}, /*hash_range=*/hash_range),
       dataset::BlockList({labels}), dataset::BlockList({make_labels})},
      /*has_header=*/true, /*delimiter=*/',');

  dataset::DatasetLoader loader(data, featurizer, /*shuffle=*/false);

  auto datasets = loader.loadAll(1000, /*verbose=*/false);

  EXPECT_EQ(datasets.at(0)->numBatches(), 1);
  EXPECT_EQ(datasets.at(1)->numBatches(), 1);
  EXPECT_EQ(datasets.at(2)->numBatches(), 1);

  return {batchToIndicesAndValues(datasets.at(0)->at(0)),
          batchToIndicesAndValues(datasets.at(1)->at(0)),
          batchToIndicesAndValues(datasets.at(2)->at(0))};
}

TEST(TextCompatTest, FullPipeline) {
  std::vector<std::string> lines = {
      "TEXT,IDS",
      R"("I like to eat apples, bannas, and mangos",1:2)",
      "\"a line with a \n newline\",3:18",
      R"(they said ""hi"" to each other,4)",
      R"(this sentence: has lots| of? strange! punctuation... but &no #commas,23:10:18)",
      "this \t has a tab,2:4:23",
      " ,50",
      "a  more standard    sentence (with STRANGE       spacing),72:3",
  };

  auto mach_index = dataset::mach::MachIndex::make(1000, 3, 100);

  for (const auto& tokenizer : getTokenizers()) {
    for (const auto& encoder : getEncoders()) {
      for (auto encoding_dim : getEncodingDims()) {
        for (auto hash_range : getHashRanges()) {
          for (auto lowercase : {true, false}) {
            auto [inputs_1, labels_1, mach_labels_1] = applyTransformations(
                std::make_shared<MockDataSource>(lines), mach_index, tokenizer,
                encoder, encoding_dim, hash_range, lowercase);

            auto [inputs_2, labels_2, mach_labels_2] = applyBlocks(
                std::make_shared<MockDataSource>(lines), mach_index, tokenizer,
                encoder, encoding_dim, hash_range, lowercase);

            ASSERT_EQ(inputs_1, inputs_2);
            ASSERT_EQ(labels_1, labels_2);
            ASSERT_EQ(mach_labels_1, mach_labels_2);
          }
        }
      }
    }
  }
}

TEST(TextCompatTest, Serialization) {
  std::vector<std::string> text = randomText(/*n_lines=*/10);

  for (const auto& tokenizer : getTokenizers()) {
    for (const auto& encoder : getEncoders()) {
      for (auto encoding_dim : getEncodingDims()) {
        for (auto hash_range : getHashRanges()) {
          for (auto lowercase : {true, false}) {
            auto [indices_1, values_1] = applyTransformation(
                tokenizer, encoder, encoding_dim, hash_range, lowercase, text);

            auto [indices_2, values_2] = applyTransformation(
                tokenizer, encoder, encoding_dim, hash_range, lowercase, text,
                /*serialize=*/true);

            ASSERT_EQ(indices_1, indices_2);
            ASSERT_EQ(values_1, values_2);
          }
        }
      }
    }
  }
}

}  // namespace thirdai::data::tests