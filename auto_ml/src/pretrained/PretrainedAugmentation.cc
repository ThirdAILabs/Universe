#include "PretrainedAugmentation.h"
#include <data/src/transformations/SpladeAugmentation.h>
#include <data/src/transformations/SpladeMachAugmentation.h>

namespace thirdai::automl {

SpladeConfig::SpladeConfig(const std::string& model_checkpoint,
                           const std::string& tokenizer_vocab,
                           std::optional<size_t> n_augmented_tokens,
                           std::optional<float> augmentation_frac,
                           bool filter_tokens, size_t batch_size,
                           bool lowercase, std::optional<size_t> feature_hashing_range)
    : _model(bolt::Model::load(model_checkpoint)),
      _tokenizer(std::make_shared<dataset::WordpieceTokenizer>(tokenizer_vocab,
                                                               lowercase)),
      _n_augmented_tokens(n_augmented_tokens),
      _augmentation_frac(augmentation_frac),
      _filter_tokens(filter_tokens),
      _batch_size(batch_size),
      _feature_hashing_range(feature_hashing_range) {}

data::TransformationPtr SpladeConfig::transformation(
    const std::string& input_col, const std::string& output_col) const {
  return std::make_shared<data::SpladeAugmentation>(
      /*input_column=*/input_col,
      /*output_column=*/output_col,
      /*model=*/_model,
      /*tokenizer=*/_tokenizer,
      /*n_augmented_tokens=*/_n_augmented_tokens,
      /*augmentation_frac=*/_augmentation_frac,
      /*filter_tokens=*/_filter_tokens,
      /*batch_size=*/_batch_size,
      /*token_offset=*/_feature_hashing_range);
}

data::TransformationPtr SpladeMachConfig::transformation(
    const std::string& input_col, const std::string& output_col) const {
  return std::make_shared<data::SpladeMachAugmentation>(
      input_col, output_col, _model, _n_hashes_per_model, _feature_hashing_range);
}

}  // namespace thirdai::automl