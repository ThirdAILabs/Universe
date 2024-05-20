#pragma once

#include <bolt/src/NER/Defaults.h>
#include <bolt/src/NER/model/NerBackend.h>
#include <bolt/src/NER/model/NerUDTModel.h>
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
#include <memory>
#include <unordered_map>

namespace thirdai::bolt::NER {

class NerBoltModel;

class NerBoltModel final : public NerModelInterface {
 public:
  std::string type() const final { return "bolt_ner"; }
  NerBoltModel(bolt::ModelPtr model, std::string tokens_column,
               std::string tags_column,
               std::unordered_map<std::string, uint32_t> tag_to_label);

  NerBoltModel(std::shared_ptr<NerBoltModel>& pretrained_model,
               std::string tokens_column, std::string tags_column,
               std::unordered_map<std::string, uint32_t> tag_to_label);

  std::vector<std::vector<std::vector<std::pair<std::string, float>>>> getTags(
      std::vector<std::vector<std::string>> tokens, uint32_t top_k) const final;

  metrics::History train(const dataset::DataSourcePtr& train_data,
                         float learning_rate, uint32_t epochs,
                         size_t batch_size,
                         const std::vector<std::string>& train_metrics,
                         const dataset::DataSourcePtr& val_data,
                         const std::vector<std::string>& val_metrics) final;

  ar::ConstArchivePtr toArchive() const final;

  std::unordered_map<std::string, uint32_t> getTagToLabel() final {
    return _tag_to_label;
  }

  static std::shared_ptr<NerBoltModel> fromArchive(const ar::Archive& archive);

  bolt::ModelPtr getBoltModel() final { return _bolt_model; }

  std::string getTokensColumn() const final { return _tokens_column; }

  std::string getTagsColumn() const final { return _tags_column; }

  NerBoltModel() = default;
  ~NerBoltModel() override = default;

 private:
  static bolt::ModelPtr initializeBoltModel(
      std::shared_ptr<NerBoltModel>& pretrained_model,
      std::unordered_map<std::string, uint32_t>& tag_to_label,
      uint32_t vocab_size);

  data::TransformationPtr getTransformations(bool inference);

  bolt::ModelPtr _bolt_model;
  std::string _tokens_column;
  std::string _tags_column;
  std::unordered_map<std::string, uint32_t> _tag_to_label;
  std::unordered_map<uint32_t, std::string> _label_to_tag_map;

  NerClassifierPtr _classifier;
  size_t _vocab_size = defaults::PRETRAINED_BOLT_VOCAB_SIZE;
};

}  // namespace thirdai::bolt::NER