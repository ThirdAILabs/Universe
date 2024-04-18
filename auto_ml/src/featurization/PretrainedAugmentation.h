#pragma once

#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/pretrained/MachPretrained.h>
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
               bool lowercase = true);

  data::TransformationPtr transformation(
      const std::string& input_col, const std::string& output_col) const final;

 private:
  bolt::ModelPtr _model;
  dataset::WordpieceTokenizerPtr _tokenizer;
  std::optional<size_t> _n_augmented_tokens;
  std::optional<float> _augmentation_frac;
  bool _filter_tokens;
  size_t _batch_size = 4096;
};

class MachPretrainedConfig final : public PretrainedAugmentation {
  MachPretrainedConfig(std::shared_ptr<MachPretrained> model,
                       size_t n_hashes_per_model)
      : _model(std::move(model)), _n_hashes_per_model(n_hashes_per_model) {}

  data::TransformationPtr transformation(
      const std::string& input_col, const std::string& output_col) const final;

 private:
  std::shared_ptr<MachPretrained> _model;
  size_t _n_hashes_per_model;
};

}  // namespace thirdai::automl