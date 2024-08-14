#pragma once

#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <archive/src/Archive.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/transformations/ner/learned_tags/CommonLearnedTags.h>
#include <data/src/transformations/ner/learned_tags/LearnedTag.h>
#include <data/src/transformations/ner/rules/Rule.h>
#include <data/src/transformations/ner/utils/TokenTagCounter.h>
#include <memory>
#include <string>
#include <variant>

namespace thirdai::automl::udt {

using TokenTags = std::vector<std::pair<std::string, float>>;
using SentenceTags = std::vector<TokenTags>;

std::shared_ptr<data::NerTokenizerUnigram> extractNerTokenizerTransform(
    const data::TransformationPtr& transform, bool is_inference);
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

  void addNerEntityToModel(
      const std::variant<std::string, data::ner::NerLearnedTag>& entity) final {
    data::ner::NerTagPtr tag;
    if (std::holds_alternative<std::string>(entity)) {
      tag = data::ner::getLearnedTagFromString(std::get<std::string>(entity));
    } else {
      tag = std::make_shared<data::ner::NerLearnedTag>(
          std::get<data::ner::NerLearnedTag>(entity));
    }

    auto supervised_unigram_transform =
        extractNerTokenizerTransform(_supervised_transform, false);

    for (const auto& existing_entities : _label_to_tag) {
      if (existing_entities->tag() == tag->tag()) {
        throw std::logic_error(
            "Entity already a part of the model. Cannot make a new model with "
            "the entity " +
            tag->tag());
      }
    }

    auto fc_layer =
        bolt::FullyConnected::cast(_model->opExecutionOrder().at(1));
    auto embedding_layer =
        bolt::Embedding::cast(_model->opExecutionOrder().at(0));

    auto input = bolt::Input::make(embedding_layer->inputDim());
    auto hidden = embedding_layer->apply(input);
    auto output_layer = bolt::FullyConnected::make(
        _label_to_tag.size() + 1, hidden->dim(), 1, "softmax", nullptr, true);

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
    _label_to_tag.push_back(tag);
    supervised_unigram_transform->addNewTagLabelEntry(tag->tag(),
                                                      _label_to_tag.size() - 1);

    supervised_unigram_transform->setTargetDim(_label_to_tag.size());
  }

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

  std::vector<data::ner::NerTagPtr> _label_to_tag;

  thirdai::data::ner::TokenTagCounterPtr _token_tag_counter;
};

}  // namespace thirdai::automl::udt