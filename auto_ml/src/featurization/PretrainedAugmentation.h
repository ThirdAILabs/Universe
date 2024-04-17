#pragma once

#include <bolt/src/nn/model/Model.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/WordpieceTokenizer.h>

namespace thirdai::automl {

class PretrainedAugmentation {
 public:
  virtual data::TransformationPtr transformation(
      const std::string& input_col, const std::string& output_col) const = 0;

  virtual ~PretrainedAugmentation() = default;
};

struct SpladeConfig final : public PretrainedAugmentation {
  SpladeConfig(const std::string& model_checkpoint,
               const std::string& tokenizer_vocab,
               std::optional<size_t> n_augmented_tokens,
               std::optional<float> augmentation_frac,
               bool filter_tokens = true, size_t batch_size = 4096,
               bool lowercase = true,
               std::optional<uint32_t> strong_sample_override = 7);

  data::TransformationPtr transformation(
      const std::string& input_col, const std::string& output_col) const final;

 private:
  bolt::ModelPtr _model;
  dataset::WordpieceTokenizerPtr _tokenizer;
  std::optional<size_t> _n_augmented_tokens;
  std::optional<float> _augmentation_frac;
  bool _filter_tokens;
  size_t _batch_size = 4096;
  std::optional<uint32_t> _strong_sample_override;
};

}  // namespace thirdai::automl