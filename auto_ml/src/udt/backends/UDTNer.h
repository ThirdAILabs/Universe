#pragma once

#include <archive/src/Archive.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/backends/cpp/NER.h>
#include <string>
#include <variant>

namespace thirdai::automl::udt {

using TokenTags = std::vector<std::pair<std::string, float>>;
using SentenceTags = std::vector<TokenTags>;

class UDTNer final : public UDTBackend {
 public:
  UDTNer(const ColumnDataTypes& data_types, const TokenTagsDataTypePtr& target,
         const std::string& target_name, const UDTNer* pretrained_model,
         const config::ArgumentMap& args);

  explicit UDTNer(const ar::Archive& archive);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options, const bolt::DistributedCommPtr& comm,
                   py::kwargs kwargs) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      py::kwargs kwargs) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class, std::optional<uint32_t> top_k,
                     const py::kwargs& kwargs) final;

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k,
                          const py::kwargs& kwargs) final;

  std::vector<std::string> listNerTags() const final {
    return _model->listNerTags();
  }

  void addNerRule(const std::string& rule_name) final {
    _model->addNerRule(rule_name);
  }

  void editNerLearnedTag(const data::ner::NerLearnedTagPtr& tag) final {
    _model->editNerLearnedTag(tag);
  }

  void addNerEntitiesToModel(
      const std::vector<std::variant<std::string, data::ner::NerLearnedTag>>&
          entities) final {
    _model->addNerEntitiesToModel(entities);
  }

  ModelPtr model() const final { return _model->model(); }

  void setModel(const ModelPtr& model) final { _model->setModel(model); }

  std::pair<std::string, std::string> sourceTargetCols() const final {
    return _model->sourceTargetCols();
  }

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::unique_ptr<UDTNer> fromArchive(const ar::Archive& archive);

  static std::string type() { return "udt_ner"; }

 private:
  std::unique_ptr<NerModel> _model;
};

}  // namespace thirdai::automl::udt