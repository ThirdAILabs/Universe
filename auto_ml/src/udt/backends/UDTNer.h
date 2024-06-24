#pragma once

#include <archive/src/Archive.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/transformations/ner/rules/Rule.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

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

  ModelPtr model() const final { return _model; }

  void setModel(const ModelPtr& model) final {
    ModelPtr& curr_model = _model;

    utils::verifyCanSetModel(curr_model, model);

    curr_model = model;
  }

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::unique_ptr<UDTNer> fromArchive(const ar::Archive& archive);

  static std::string type() { return "udt_ner"; }

 private:
  data::LoaderPtr getDataLoader(const dataset::DataSourcePtr& data,
                                size_t batch_size, bool shuffle) const;

  std::vector<SentenceTags> predictTags(
      const std::vector<std::string>& sentences, bool sparse_inference,
      uint32_t top_k, float o_threshold);

  struct NerOptions;

  static NerOptions fromPretrained(const UDTNer* pretrained_model);

  static NerOptions fromScratch(const config::ArgumentMap& args);

  bolt::ModelPtr _model;

  data::ner::RulePtr _rule;

  data::TransformationPtr _supervised_transform;
  data::TransformationPtr _inference_transform;

  data::OutputColumnsList _bolt_inputs;

  std::string _tokens_column;
  std::string _tags_column;

  std::vector<std::string> _label_to_tag;

  std::unordered_set<std::string> _ignored_tags;
};

}  // namespace thirdai::automl::udt