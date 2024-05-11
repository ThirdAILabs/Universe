#include "NerUnigramModel.h"
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/ner/NerTokenizationUnigram.h>

namespace thirdai::bolt {
NerUnigramModel::NerUnigramModel(
    bolt::ModelPtr model, std::string tokens_column, std::string tags_column,
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers)
    : _bolt_model(std::move(model)),
      _tokens_column(std::move(tokens_column)),
      _tags_column(std::move(tags_column)),
      _target_word_tokenizers(std::move(target_word_tokenizers)) {
  auto train_transformation = thirdai::data::NerTokenizerUnigram(
      _tokens_column, _featurized_sentence_column, _tags_column, _fhr,
      _dyadic_num_intervals, _target_word_tokenizers);

  auto inference_transformation = thirdai::data::NerTokenizerUnigram(
      _tokens_column, _featurized_sentence_column, std::nullopt, _fhr,
      _dyadic_num_intervals, _target_word_tokenizers);

  _train_transforms = data::Pipeline::make(
      {std::make_shared<thirdai::data::NerTokenizerUnigram>(
          train_transformation)});

  _inference_transforms = data::Pipeline::make(
      {std::make_shared<thirdai::data::NerTokenizerUnigram>(
          inference_transformation)});
}
}  // namespace thirdai::bolt