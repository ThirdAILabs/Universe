#pragma once

#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/pretrained/SpladeMach.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/WordpieceTokenizer.h>

namespace thirdai::automl {

class PretrainedAugmentation {
 public:
  virtual data::TransformationPtr transformation(
      const std::string& input_col, const std::string& output_col) const = 0;

  virtual bool useByDefault() const = 0;

  virtual bool containsOffset() const = 0;

  virtual std::optional<size_t> spladeInputRange() const = 0;

  virtual ~PretrainedAugmentation() = default;
};

class SpladeConfig final : public PretrainedAugmentation {
 public:
  SpladeConfig(const std::string& model_checkpoint,
               const std::string& tokenizer_vocab,
               std::optional<size_t> n_augmented_tokens,
               std::optional<float> augmentation_frac,
               bool filter_tokens = true, size_t batch_size = 4096,
               bool lowercase = true, std::optional<size_t> feature_hashing_range = std::nullopt);

  data::TransformationPtr transformation(
      const std::string& input_col, const std::string& output_col) const final;

  bool useByDefault() const final { return false; }
 bool containsOffset() const final {
    return _feature_hashing_range.has_value();
}

  std::optional<size_t> spladeInputRange() const final {
    if(_feature_hashing_range){
      return _model->outputs()[0]->dim(); 
    }else{
      return 0;
    }
  }

 private:
  bolt::ModelPtr _model;
  dataset::WordpieceTokenizerPtr _tokenizer;
  std::optional<size_t> _n_augmented_tokens;
  std::optional<float> _augmentation_frac;
  bool _filter_tokens;
  size_t _batch_size = 4096;
  std::optional<size_t> _feature_hashing_range;
};

class SpladeMachConfig final : public PretrainedAugmentation {
 public:
  SpladeMachConfig(std::shared_ptr<SpladeMach> model, size_t n_hashes_per_model, std::optional<size_t> feature_hashing_range = std::nullopt)
      : _model(std::move(model)), _n_hashes_per_model(n_hashes_per_model), _feature_hashing_range(feature_hashing_range) {}

  data::TransformationPtr transformation(
      const std::string& input_col, const std::string& output_col) const final;

  bool useByDefault() const final { return true; }

bool containsOffset() const final {
    return _feature_hashing_range.has_value();
}

  std::optional<size_t> spladeInputRange() const final { 
    if(_feature_hashing_range){
      return _model->sumOutputDimensions();
    }
    return 0;
}

 private:
  std::shared_ptr<SpladeMach> _model;
  size_t _n_hashes_per_model;
  std::optional<size_t> _feature_hashing_range;
};

}  // namespace thirdai::automl