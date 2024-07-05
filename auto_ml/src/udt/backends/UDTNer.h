#pragma once

#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <archive/src/Archive.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/transformations/ner/rules/Rule.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::automl::udt {

using TokenTags = std::vector<std::pair<std::string, float>>;
using SentenceTags = std::vector<TokenTags>;

std::shared_ptr<data::NerTokenizerUnigram> extractInputTransform(
    const data::TransformationPtr& transform);

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

  void addNerRule(const std::shared_ptr<data::ner::Pattern>& new_rule) final {
    if (_rule != nullptr) {
      for (const auto& existing_rule : _rule->entities()) {
        if (existing_rule == new_rule->entities()[0]) {
          throw std::logic_error(
              "Entity already present. Cannot add a new rule for the entity " +
              existing_rule);
        }
      }
      _rule->addRule(new_rule);
    } else {
      std::vector<data::ner::RulePtr> rule_vector = {new_rule};
      _rule = std::make_shared<data::ner::RuleCollection>(rule_vector);
    }
    auto supervised_unigram_transform =
        extractInputTransform(_supervised_transform);
    supervised_unigram_transform->addNewTagLabelEntry(new_rule->entities()[0],
                                                      0);
  }

  void addNewEntityToModel(const std::string& entity) final {
    auto supervised_unigram_transform =
        extractInputTransform(_supervised_transform);
    for (const auto& existing_entities : _label_to_tag) {
      if (existing_entities == entity) {
        throw std::logic_error(
            "Entity already a part of the model. Cannot make a new model with "
            "the entity " +
            entity);
      }
    }

    _label_to_tag.push_back(entity);

    auto fc_layer =
        bolt::FullyConnected::cast(_model->opExecutionOrder().at(1));
    auto embedding_layer =
        bolt::Embedding::cast(_model->opExecutionOrder().at(0));

    auto input = bolt::Input::make(embedding_layer->inputDim());
    // auto emb_op = bolt::Embedding::make(
    //     embedding_layer->dim(), embedding_layer->inputDim(), "relu", true);

    auto hidden = embedding_layer->apply(input);

    auto output_layer = bolt::FullyConnected::make(
        _label_to_tag.size(), hidden->dim(), 1, "softmax", nullptr, true);

    {
      const float* weight_start = output_layer->weightsPtr();
      int weight_size = output_layer->dim() * output_layer->inputDim();
      std::vector<float> new_weights(weight_start, weight_start + weight_size);

      const float* biases_start = output_layer->weightsPtr();
      int biases_size = output_layer->dim() * output_layer->inputDim();
      std::vector<float> new_biases(biases_start, biases_start + biases_size);

      const float* old_weights = fc_layer->weightsPtr();
      const float* old_biases = fc_layer->biasesPtr();
      for (size_t i = 0; i < fc_layer->dim() * fc_layer->inputDim(); i++) {
        new_weights[i] = old_weights[i];
      }
      for (size_t i = 0; i < fc_layer->dim(); ++i) {
        new_biases[i] = old_biases[i];
      }

      output_layer->setWeights(new_weights.data());
      output_layer->setBiases(new_biases.data());
    }

    auto output = output_layer->apply(hidden);

    auto labels = bolt::Input::make(output->dim());
    auto loss = bolt::CategoricalCrossEntropy::make(output, labels);

    _model = bolt::Model::make({input}, {output}, {loss});
  }

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

  bool _ignore_rule_tags;
};

}  // namespace thirdai::automl::udt