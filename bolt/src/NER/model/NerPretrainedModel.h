#pragma once

#include "NerClassifier.h"
#include <bolt/src/NER/model/NerBackend.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/text_generation/GenerativeModel.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/DyadicInterval.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/DataSource.h>
#include <unordered_map>

namespace thirdai::bolt {

class NerPretrainedModel final : public NerModelInterface {
 public:
  std::string type() const final { return "pretrained_ner"; }
  explicit NerPretrainedModel(
      bolt::ModelPtr model, std::string tokens_column, std::string tags_column,
      std::unordered_map<std::string, uint32_t> tag_to_label);

  explicit NerPretrainedModel(
      std::string& pretrained_model_path, std::string tokens_column,
      std::string tags_column,
      std::unordered_map<std::string, uint32_t> tag_to_label);

  std::vector<PerTokenListPredictions> getTags(
      std::vector<std::vector<std::string>> tokens, uint32_t top_k) const final;

  metrics::History train(
      const dataset::DataSourcePtr& train_data, float learning_rate,
      uint32_t epochs, size_t batch_size,
      const std::vector<std::string>& train_metrics,
      const dataset::DataSourcePtr& val_data,
      const std::vector<std::string>& val_metrics) const final;

  std::unordered_map<std::string, uint32_t> getTagToLabel() const final {
    return _tag_to_label;
  }

  ar::ConstArchivePtr toArchive() const final;

  static std::shared_ptr<NerPretrainedModel> fromArchive(
      const ar::Archive& archive);

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<NerPretrainedModel> load_stream(
      std::istream& input_stream);

  ~NerPretrainedModel() override = default;

 private:
  data::PipelinePtr getTransformations(bool inference);

  static bolt::ModelPtr getBoltModel(
      std::string& pretrained_model_path,
      std::unordered_map<std::string, uint32_t> tag_to_label,
      uint32_t vocab_size);

  bolt::ModelPtr _bolt_model;
  std::string _tokens_column;
  std::string _tags_column;
  std::unordered_map<std::string, uint32_t> _tag_to_label;

  NerClassifierPtr _classifier;

  const size_t _vocab_size = 50257;
};

}  // namespace thirdai::bolt