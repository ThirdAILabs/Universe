#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/encodings/categorical/CategoricalMultiLabel.h>
#include <dataset/src/encodings/text/PairGram.h>
#include <cstdint>
#include <string>
#include <tuple>
namespace thirdai::dataset {

class MultiLabelTextDatasetProcessor {
 public: 
  explicit MultiLabelTextDatasetProcessor(uint32_t n_classes) {
    buildBatchProcessors(n_classes);
  }

  std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadFromFile(const std::string& filename, bool shuffle) {
    StreamingGenericDatasetLoader loader(filename, _processor, /* batch_size= */ 2048, shuffle);
    return loader.loadInMemory();
  }

  bolt::BoltVector fromTokens(const std::vector<uint32_t>& tokens) {
    auto sentence = tokensToSentence(tokens);
    // The following step must be separate from the above
    // because we need to keep the sentence in scope and alive.
    std::vector<std::string_view> sample = {std::string_view(sentence.data(), sentence.size())};

    bolt::BoltVector input_vector;
    auto exception = _inference_processor->makeInputVector(sample, input_vector);
    if (exception) {
      std::rethrow_exception(exception);
    }
    return input_vector;
  }

 private:
  void buildBatchProcessors(uint32_t n_classes) {
    _processor = std::make_shared<dataset::GenericBatchProcessor>(
        buildInputBlocks(/* for_single_inference= */ false), 
        buildLabelBlocks(/* for_single_inference= */ false, n_classes),
        /* has_header= */ false, /* delimiter= */ '\t');

    _inference_processor = std::make_shared<dataset::GenericBatchProcessor>(
        buildInputBlocks(/* for_single_inference= */ true), 
        buildLabelBlocks(/* for_single_inference= */ true),
        /* has_header= */ false, /* delimiter= */ '\t');
  }

  static std::vector<dataset::BlockPtr> buildInputBlocks(bool for_single_inference) {
    auto pairgram_encoding =
        std::make_shared<dataset::PairGram>(/* dim= */ 100000);
    uint32_t column = for_single_inference ? 0 : 1;
    return {std::make_shared<dataset::TextBlock>(
        column, pairgram_encoding)};
  }

  static std::vector<dataset::BlockPtr> buildLabelBlocks(bool for_single_inference, uint32_t n_classes=0) {
    if (!for_single_inference && n_classes == 0) {
      throw std::invalid_argument("buildLabelBlocks: Must pass n_classes if not for single inference.");
    }
    if (for_single_inference) {
      return {};
    }
    auto multi_label_encoding =
        std::make_shared<dataset::CategoricalMultiLabel>(
            /* n_classes= */ n_classes, /* delimiter= */ ',');
    return {std::make_shared<dataset::CategoricalBlock>(
        /* col= */ 0, /* encoding= */ multi_label_encoding)};
  }

  static std::string tokensToSentence(const std::vector<uint32_t>& tokens) {
    std::stringstream sentence_ss;
    for (uint32_t i = 0; i < tokens.size(); i++) {
      if (i > 0) {
        sentence_ss << ' ';
      }
      sentence_ss << tokens[i];
    }
    return sentence_ss.str();
  }

  GenericBatchProcessorPtr _processor;
  GenericBatchProcessorPtr _inference_processor;
};

} // namespace thirdai::dataset