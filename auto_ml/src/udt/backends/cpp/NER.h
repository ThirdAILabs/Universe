#pragma once

#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/trainer/DistributedComm.h>
#include <archive/src/Archive.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/TrainOptions.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/Loader.h>
#include <data/src/transformations/ner/NerTokenizationUnigram.h>
#include <data/src/transformations/ner/learned_tags/CommonLearnedTags.h>
#include <data/src/transformations/ner/learned_tags/LearnedTag.h>
#include <data/src/transformations/ner/rules/CommonPatterns.h>
#include <data/src/transformations/ner/rules/Rule.h>
#include <data/src/transformations/ner/utils/TagTracker.h>
#include <data/src/transformations/ner/utils/TokenLabelCounter.h>
#include <memory>
#include <string>
#include <variant>

namespace thirdai::automl::udt {

using TokenTags = std::vector<std::pair<std::string, float>>;
using SentenceTags = std::vector<TokenTags>;

using Predictions =
    std::pair<std::vector<SentenceTags>,
              std::vector<std::vector<std::pair<size_t, size_t>>>>;

std::shared_ptr<data::NerTokenizerUnigram> extractNerTokenizerTransform(
    const data::TransformationPtr& transform, bool is_inference);

struct LabeledEntity {
  std::string label;
  std::string text;
  size_t start;
  size_t end;
};

class NerModel {
 public:
  NerModel(const ColumnDataTypes& data_types,
           const TokenTagsDataTypePtr& target, const std::string& target_name,
           const NerModel* pretrained_model, const config::ArgumentMap& args);

  explicit NerModel(const ar::Archive& archive);

  void train(const std::string& filename, float learning_rate, uint32_t epochs);

  std::unordered_map<std::string, std::vector<float>> train(
      const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
      const std::vector<std::string>& train_metrics,
      const dataset::DataSourcePtr& val_data,
      const std::vector<std::string>& val_metrics,
      const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
      TrainOptions options, const bolt::DistributedCommPtr& comm);

  std::unordered_map<std::string, std::vector<float>> evaluate(
      const dataset::DataSourcePtr& data,
      const std::vector<std::string>& metrics, bool sparse_inference,
      bool verbose);

  std::vector<std::vector<LabeledEntity>> predict(
      const std::vector<std::string>& sentences);

  std::pair<std::vector<SentenceTags>,
            std::vector<std::vector<std::pair<size_t, size_t>>>>
  predictTags(const std::vector<std::string>& sentences, bool sparse_inference,
              uint32_t top_k, float o_threshold, bool as_unicode);

  std::vector<std::string> listNerTags() const {
    return _tag_tracker->listNerTags();
  }

  void addNerRule(const std::string& rule_name) {
    if (_rule != nullptr) {
      auto rule_collection =
          std::dynamic_pointer_cast<data::ner::RuleCollection>(_rule);

      for (const auto& existing_rule : rule_collection->entities()) {
        if (existing_rule == rule_name) {
          throw std::logic_error(
              "Entity already present. Cannot add a new rule for the entity " +
              existing_rule);
        }
      }
      rule_collection->addRule(data::ner::getRuleForEntity(rule_name));

    } else {
      std::vector<data::ner::RulePtr> rule_vector = {
          data::ner::getRuleForEntity(rule_name)};
      _rule = std::make_shared<data::ner::RuleCollection>(rule_vector);
    }

    // if tag is not present in the tracker -> new tag and assign a label
    // if tag is present -> old tag, do not update label
    if (!_tag_tracker->tagExists(rule_name)) {
      _tag_tracker->addTag(data::ner::getLearnedTagFromString(rule_name),
                           /*add_new_label=*/false);
    }
  }

  void editNerLearnedTag(const data::ner::NerLearnedTagPtr& tag) {
    _tag_tracker->editTag(tag);
  }

  void addNerEntitiesToModel(
      const std::vector<std::variant<std::string, data::ner::NerLearnedTag>>&
          entities) {
    std::vector<data::ner::NerTagPtr> tags;

    {
      // initialize the tags vector.
      for (const auto& entity : entities) {
        if (std::holds_alternative<std::string>(entity)) {
          tags.emplace_back(data::ner::getLearnedTagFromString(
              std::get<std::string>(entity)));
        } else {
          tags.emplace_back(std::make_shared<data::ner::NerLearnedTag>(
              std::get<data::ner::NerLearnedTag>(entity)));
        }
      }

      // check that none of the entities provided already exists
      for (const auto& tag : tags) {
        if (_tag_tracker->tagExists(tag->tag())) {
          throw std::logic_error("Cannot add entity " + tag->tag() +
                                 "to the model. Entity already exists");
        }
      }
    }

    auto supervised_unigram_transform =
        extractNerTokenizerTransform(_supervised_transform, false);

    auto fc_layer =
        bolt::FullyConnected::cast(_model->opExecutionOrder().at(1));
    auto embedding_layer =
        bolt::Embedding::cast(_model->opExecutionOrder().at(0));

    auto input = bolt::Input::make(embedding_layer->inputDim());
    auto hidden = embedding_layer->apply(input);
    auto output_layer = bolt::FullyConnected::make(
        _tag_tracker->numLabels() + tags.size(), hidden->dim(),
        fc_layer->getSparsity(),
        bolt::activationFunctionToStr(
            fc_layer->kernel()->getActivationFunction()),
        nullptr, fc_layer->kernel()->useBias());

    {
      // copy the weights and biases to the new model
      auto& new_weights = output_layer->kernel()->weights();
      const auto& old_weights = fc_layer->kernel()->weights();
      std::copy(old_weights.begin(), old_weights.end(), new_weights.begin());

      auto& new_bias = output_layer->kernel()->biases();
      const auto& old_bias = fc_layer->kernel()->biases();
      std::copy(old_bias.begin(), old_bias.end(), new_bias.begin());
    }

    auto output = output_layer->apply(hidden);
    auto labels = bolt::Input::make(output->dim());
    auto loss = bolt::CategoricalCrossEntropy::make(output, labels);

    _model = bolt::Model::make({input}, {output}, {loss});

    for (const auto& tag : tags) {
      _tag_tracker->addTag(tag, /*add_new_label=*/true);
    }
  }

  bolt::ModelPtr model() const { return _model; }

  void setModel(const bolt::ModelPtr& model) {
    bolt::ModelPtr& curr_model = _model;

    utils::verifyCanSetModel(curr_model, model);

    curr_model = model;
  }

  std::pair<std::string, std::string> sourceTargetCols() const;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const;

  static std::unique_ptr<NerModel> fromArchive(const ar::Archive& archive);

  static std::string type() { return "udt_ner"; }

  const auto& getTokensColumn() const { return _tokens_column; }

  static std::unique_ptr<NerModel> load(const std::string& path);

  void save(const std::string& path) const;

 private:
  NerModel() {}

  data::LoaderPtr getDataLoader(const dataset::DataSourcePtr& data,
                                size_t batch_size, bool shuffle) const;

  struct NerOptions;

  static NerOptions fromPretrained(const NerModel* pretrained_model);

  static NerOptions fromScratch(const config::ArgumentMap& args);

  friend class cereal::access;

  template <class Archive>
  void save(Archive& archive, uint32_t version) const;

  template <class Archive>
  void load(Archive& archive, uint32_t version);

  bolt::ModelPtr _model;

  data::ner::RulePtr _rule;

  data::TransformationPtr _supervised_transform;
  data::TransformationPtr _inference_transform;

  data::OutputColumnsList _bolt_inputs;

  std::string _tokens_column;
  std::string _tags_column;

  data::ner::utils::TagTrackerPtr _tag_tracker;
};

}  // namespace thirdai::automl::udt