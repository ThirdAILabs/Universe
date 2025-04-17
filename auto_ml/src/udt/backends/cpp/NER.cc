#include "NER.h"
#include <bolt/src/NER/model/NerClassifier.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/udt/Defaults.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringSplitOnWhiteSpace.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/ner/NerTokenizationUnigram.h>
#include <data/src/transformations/ner/learned_tags/LearnedTag.h>
#include <data/src/transformations/ner/rules/CommonPatterns.h>
#include <data/src/transformations/ner/rules/Rule.h>
#include <data/src/transformations/ner/utils/TagTracker.h>
#include <data/src/transformations/ner/utils/utils.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

namespace thirdai::automl::udt {

bolt::ModelPtr buildModel(uint32_t input_dim, uint32_t emb_dim,
                          uint32_t output_dim,
                          const bolt::EmbeddingPtr& pretrained_emb) {
  auto input = bolt::Input::make(input_dim);

  bolt::EmbeddingPtr emb_op;
  if (pretrained_emb) {
    emb_op = bolt::Embedding::make(
        pretrained_emb->dim(), pretrained_emb->inputDim(),
        bolt::activationFunctionToStr(pretrained_emb->activation()),
        pretrained_emb->useBias());
    emb_op->setEmbeddings(pretrained_emb->embeddingsPtr());
    emb_op->setBiases(pretrained_emb->biasesPtr());
  } else {
    emb_op = bolt::Embedding::make(emb_dim, input_dim, "relu",
                                   /* bias= */ true);
  }

  auto hidden = emb_op->apply(input);

  auto output =
      bolt::FullyConnected::make(output_dim, hidden->dim(), 1, "softmax",
                                 /* sampling= */ nullptr, /* use_bias= */ true)
          ->apply(hidden);

  auto labels = bolt::Input::make(output_dim);
  auto loss = bolt::CategoricalCrossEntropy::make(output, labels);

  return bolt::Model::make({input}, {output}, {loss});
}

data::TransformationPtr makeTransformation(
    bool inference, const std::string& tags_column,
    const std::string& tokens_column,
    const data::ner::utils::TagTrackerPtr& tag_tracker, size_t input_dim,
    uint32_t dyadic_num_intervals,
    const std::vector<dataset::TextTokenizerPtr>& target_word_tokenizers,
    const std::optional<data::FeatureEnhancementConfig>& feature_config) {
  std::optional<std::string> target_column = tags_column;
  if (inference) {
    target_column = std::nullopt;
  }

  auto transform = data::Pipeline::make();
  if (!inference) {
    transform = transform->then(std::make_shared<data::StringToStringArray>(
        tokens_column, tokens_column, ' ', std::nullopt));
    transform = transform->then(std::make_shared<data::StringToStringArray>(
        target_column.value(), target_column.value(), ' ', std::nullopt));
  }

  transform = transform
                  ->then(std::make_shared<data::NerTokenizerUnigram>(
                      /*tokens_column=*/tokens_column,
                      /*featurized_sentence_column=*/NER_FEATURIZED_SENTENCE,
                      /*target_column=*/target_column,
                      /*dyadic_num_intervals=*/dyadic_num_intervals,
                      /*target_word_tokenizers=*/target_word_tokenizers,
                      /*feature_enhancement_config=*/feature_config,
                      /*tag_tracker=*/tag_tracker))
                  ->then(std::make_shared<data::TextTokenizer>(
                      /*input_column=*/NER_FEATURIZED_SENTENCE,
                      /*output_indices=*/NER_FEATURIZED_SENTENCE,
                      // TODO(Any): Should we specify output_values so that
                      // tokens are deduplicated?
                      /*output_values=*/std::nullopt,
                      /*tokenizer=*/
                      std::make_shared<dataset::NaiveSplitTokenizer>(
                          dataset::NaiveSplitTokenizer()),
                      /*encoder=*/std::make_shared<dataset::NGramEncoder>(1),
                      /*lowercase=*/false, /*dim=*/input_dim));

  return transform;
}

std::string tokensColumn(ColumnDataTypes data_types,
                         const std::string& target) {
  if (!data_types.count(target)) {
    throw std::invalid_argument(
        "Target column must be specified in the data types.");
  }
  data_types.erase(target);

  if (data_types.size() != 1 || !asText(data_types.begin()->second)) {
    throw std::invalid_argument(
        "Token classification models must have a single text input.");
  }

  return data_types.begin()->first;
}

std::vector<data::ner::NerTagPtr> mapTagsToLabels(
    const std::string& default_tag,
    const std::vector<std::variant<std::string, data::ner::NerLearnedTag>>&
        tags) {
  /*
   * Constructs a vector of NerLearnedTags.
   * The vector represents the tags as they will appear in the model.
   *
   * Process:
   * 1. If the tag is an instance of NerLearnedTag -> directly added to the
   * output model.
   * 2. If the tag is a string, it is converted to NerLearnedTag and added to
   * the model.
   */

  std::vector<data::ner::NerTagPtr> model_tags = {
      data::ner::getLearnedTagFromString(default_tag)};

  for (const auto& tag : tags) {
    if (std::holds_alternative<data::ner::NerLearnedTag>(tag)) {
      // Direct inclusion for explicitly defined NerLearnedTag objects.
      model_tags.push_back(std::make_shared<data::ner::NerLearnedTag>(
          std::get<data::ner::NerLearnedTag>(tag)));
    } else {
      auto tag_string = std::get<std::string>(tag);
      model_tags.push_back(data::ner::getLearnedTagFromString(tag_string));
    }
  }
  return model_tags;
}

std::shared_ptr<data::NerTokenizerUnigram> extractNerTokenizerTransform(
    const data::TransformationPtr& transform, bool is_inference) {
  if (auto pipeline = std::dynamic_pointer_cast<data::Pipeline>(transform)) {
    if (pipeline->transformations().empty()) {
      return nullptr;
    }

    return std::dynamic_pointer_cast<data::NerTokenizerUnigram>(
        is_inference ? pipeline->transformations().at(0)
                     : pipeline->transformations().at(2));
  }

  return nullptr;
}

void normalizeScores(TokenTags& tags) {
  float squared_sum = 0.0;
  for (const auto& [_, score] : tags) {
    squared_sum += score * score;
  }

  // avoid division by zero
  const float epsilon = 1e-10;
  float norm = std::sqrt(std::max(squared_sum, epsilon));

  for (auto& [_, score] : tags) {
    score /= norm;
  }
}

struct NerModel::NerOptions {
  uint32_t input_dim;
  int32_t emb_dim;
  uint32_t dyadic_num_intervals;
  std::vector<dataset::TextTokenizerPtr> target_tokenizers;
  std::optional<data::FeatureEnhancementConfig> feature_config;
  bolt::EmbeddingPtr pretrained_emb;
};

NerModel::NerOptions NerModel::fromPretrained(
    const NerModel* pretrained_model) {
  NerOptions options;

  options.pretrained_emb =
      bolt::Embedding::cast(pretrained_model->_model->opExecutionOrder().at(0));
  if (!options.pretrained_emb) {
    throw std::invalid_argument("Invalid pretrained model for NER task.");
  }

  options.input_dim = options.pretrained_emb->inputDim();
  options.emb_dim = options.pretrained_emb->dim();

  auto input_transform = extractNerTokenizerTransform(
      pretrained_model->_supervised_transform, /*is_inference=*/false);
  if (!input_transform) {
    throw std::invalid_argument("Invalid pretrained model for NER task.");
  }

  options.dyadic_num_intervals =
      input_transform->processor().nDyadicIntervals();

  options.target_tokenizers =
      input_transform->processor().targetWordTokenizers();

  options.feature_config = input_transform->processor().featureConfig();

  return options;
}

NerModel::NerOptions NerModel::fromScratch(const config::ArgumentMap& args) {
  NerOptions options;

  options.input_dim =
      args.get<uint32_t>("input_dim", "integer", defaults::FEATURE_HASH_RANGE);
  if (args.contains("fhr")) {
    options.input_dim = args.get<uint32_t>("fhr", "integer");
  }

  options.emb_dim = args.get<uint32_t>("embedding_dimension", "integer",
                                       defaults::NER_EMB_DIM);

  options.dyadic_num_intervals = defaults::NER_DYADIC_INTERVALS;

  options.target_tokenizers = args.get<std::vector<dataset::TextTokenizerPtr>>(
      "target_tokenizers", "List[Tokenizer]",
      {{std::make_shared<dataset::NaiveSplitTokenizer>(),
        std::make_shared<dataset::CharKGramTokenizer>(4)}});

  options.feature_config = args.get<data::FeatureEnhancementConfig>(
      "feature_config", "FeatureConfig", data::FeatureEnhancementConfig());

  options.pretrained_emb = nullptr;

  return options;
}

NerModel::NerModel(const ColumnDataTypes& data_types,
                   const TokenTagsDataTypePtr& target,
                   const std::string& target_name,
                   const NerModel* pretrained_model,
                   const config::ArgumentMap& args)
    : _bolt_inputs({data::OutputColumns(NER_FEATURIZED_SENTENCE)}),
      _tokens_column(tokensColumn(data_types, target_name)),
      _tags_column(target_name) {
  NerOptions options;
  if (pretrained_model) {
    options = fromPretrained(pretrained_model);
  } else {
    options = fromScratch(args);
  }

  std::vector<data::ner::NerTagPtr> model_tags;

  model_tags = mapTagsToLabels(target->default_tag, target->tags);

  if (args.get<bool>("use_token_tag_counter", "bool", false)) {
    _tag_tracker = data::ner::utils::TagTracker::make(
        model_tags, {},
        args.get<uint32_t>("token_counter_bins", "uint32_t", 10));
  } else {
    _tag_tracker = data::ner::utils::TagTracker::make(model_tags, {});
  }

  _model = buildModel(options.input_dim, options.emb_dim,
                      _tag_tracker->numLabels(), options.pretrained_emb);

  _supervised_transform = makeTransformation(
      /*inference=*/false, /*tags_column=*/_tags_column,
      /*tokens_column=*/_tokens_column, _tag_tracker,
      /*input_dim=*/options.input_dim,
      /*dyadic_num_intervals=*/options.dyadic_num_intervals,
      /*target_word_tokenizers=*/options.target_tokenizers,
      /*feature_config=*/options.feature_config);

  _inference_transform = makeTransformation(
      /*inference=*/true, /*tags_column=*/_tags_column,
      /*tokens_column=*/_tokens_column, _tag_tracker,
      /*input_dim=*/options.input_dim,
      /*dyadic_num_intervals=*/options.dyadic_num_intervals,
      /*target_word_tokenizers=*/options.target_tokenizers,
      /*feature_config=*/options.feature_config);

  std::cout << "Initialized a UniversalDeepTransformer for Token Classification"
            << std::endl;
}

std::unordered_map<std::string, std::vector<float>> NerModel::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    TrainOptions options, const bolt::DistributedCommPtr& comm) {
  bolt::Trainer trainer(_model);

  auto history = trainer.train_with_data_loader(
      /*train_data_loader=*/getDataLoader(data, options.batchSize(),
                                          /*shuffle=*/true),
      /*learning_rate=*/learning_rate, /*epochs=*/epochs,
      /*max_in_memory_batches=*/options.max_in_memory_batches,
      /*train_metrics=*/
      bolt::metrics::fromMetricNames(_model, train_metrics, "train_"),
      /*validation_data_loader=*/
      val_data
          ? getDataLoader(val_data, defaults::BATCH_SIZE, /*shuffle=*/false)
          : nullptr,
      /*validation_metrics=*/
      bolt::metrics::fromMetricNames(_model, val_metrics, "val_"),
      /*steps_per_validation=*/options.steps_per_validation,
      /*use_sparsity_in_validation=*/options.sparse_validation,
      /*callbacks=*/callbacks,
      /*autotune_rehash_rebuild*/ false,
      /*verbose=*/options.verbose,
      /*logging_interval=*/options.logging_interval,
      /*comm=*/comm);

  return history;
}

std::unordered_map<std::string, std::vector<float>> NerModel::evaluate(
    const dataset::DataSourcePtr& data, const std::vector<std::string>& metrics,
    bool sparse_inference, bool verbose) {
  bolt::Trainer trainer(_model);

  auto history = trainer.validate_with_data_loader(
      getDataLoader(data, defaults::BATCH_SIZE, /*shuffle=*/false),
      bolt::metrics::fromMetricNames(_model, metrics, "val_"), sparse_inference,
      verbose);

  return history;
}

std::pair<std::vector<SentenceTags>,
          std::vector<std::vector<std::pair<size_t, size_t>>>>
NerModel::predictTags(const std::vector<std::string>& sentences,
                      bool sparse_inference, uint32_t top_k, float o_threshold,
                      bool as_unicode) {
  std::vector<std::vector<std::string>> tokens;
  std::vector<SentenceTags> output_tags(sentences.size());

  auto sentence_column =
      data::ValueColumn<std::string>::make(std::vector<std::string>{sentences});
  auto data = data::ColumnMap({{_tokens_column, sentence_column}});

  auto split_sentence_transform =
      data::StringSplitOnWhiteSpace(_tokens_column, _tokens_column, as_unicode);

  auto tokenized_sentences = split_sentence_transform.applyStateless(data);

  data::ArrayColumnBasePtr<std::string> tokens_columns =
      tokenized_sentences.getArrayColumn<std::string>(_tokens_column);

  const data::ArrayColumnBasePtr<std::pair<size_t, size_t>> offsets =
      tokenized_sentences.getArrayColumn<std::pair<size_t, size_t>>(
          _tokens_column + "_offsets");

  // Get and stores offsets from transformation
  std::vector<std::vector<std::pair<size_t, size_t>>> token_offsets(
      tokenized_sentences.numRows());

  for (uint32_t i = 0; i < tokenized_sentences.numRows(); i++) {
    auto offset_row = offsets->row(i);
    token_offsets[i].reserve(offset_row.size());
    for (const auto& offset : offset_row) {
      token_offsets[i].emplace_back(offset);
    }
  }

  auto token_ptr =
      std::dynamic_pointer_cast<data::ArrayColumn<std::string>>(tokens_columns);
  if (token_ptr) {
    tokens = token_ptr->data();
  } else {
    throw std::logic_error("Cannot convert sentences to tokens.");
  }

  auto featurized = _inference_transform->applyStateless(tokenized_sentences);
  auto tensors =
      data::toTensorBatches(featurized, _bolt_inputs, defaults::BATCH_SIZE);

  size_t sentence_index = 0;
  size_t token_index = 0;

  std::vector<SentenceTags> rule_results;
  if (_rule) {
    rule_results = _rule->applyBatch(tokens);
  }

  for (const auto& batch : tensors) {
    auto scores = _model->forward(batch, sparse_inference).at(0);

    for (size_t i = 0; i < scores->batchSize(); i++) {
      while (token_index == tokens[sentence_index].size()) {
        sentence_index++;
        token_index = 0;
      }

      TokenTags tags;
      TokenTags rule_tags;
      std::unordered_map<std::string, float> tags_to_score;

      if (_rule && !rule_results.at(sentence_index).at(token_index).empty()) {
        rule_tags = rule_results.at(sentence_index).at(token_index);
        for (const auto& [tag, score] : rule_tags) {
          tags_to_score[tag] = score;
        }
      }
      // this block generates the model tags
      {
        TokenTags model_tags;
        auto top_labels = scores->getVector(i).topKNeurons(top_k + 1);

        while (!top_labels.empty()) {
          float score = top_labels.top().first;

          auto tag = _tag_tracker->labelToTag(top_labels.top().second);
          top_labels.pop();
          model_tags.emplace_back(tag->tag(), score);
        }

        bolt::NER::applyPunctAndStopWordFilter(
            tokens[sentence_index][token_index], model_tags,
            _tag_tracker->labelToTag(0)->tag());

        for (const auto& [tag, score] : model_tags) {
          // if the default tag is the top prediction of the model but rules
          // have predicted a tag, then we do not consider the model prediction
          if (tag == _tag_tracker->labelToTag(0)->tag() && !rule_tags.empty()) {
            continue;
          }
          if (tags_to_score.find(tag) == tags_to_score.end()) {
            tags_to_score[tag] = score;
          } else {
            tags_to_score[tag] += score;
          }
        }
      }

      // this block generates the final tags
      {
        for (const auto& [tag, score] : tags_to_score) {
          tags.emplace_back(tag, score);
        }

        assert(tags.size() >= 1);

        std::sort(tags.begin(), tags.end(), [](const auto& a, const auto& b) {
          return b.second > a.second;
        });

        // if the number of labels in the model is 1, we do not have to
        // reverse
        if (tags.size() > 1) {
          // If the default tag is the top prediction but has a score <
          // 0.9 then using the next top prediction improves accuracy.
          float second_highest_tag_act =
              top_k > 0 ? tags[tags.size() - 2].second : 0;

          if (tags.back().first == _tag_tracker->labelToTag(0)->tag() &&
              tags.back().second < o_threshold &&
              second_highest_tag_act > 0.05) {
            tags.pop_back();
            std::reverse(tags.begin(), tags.end());
          } else {
            std::reverse(tags.begin(), tags.end());
            tags.pop_back();
          }
          while (tags.size() > top_k) {
            tags.pop_back();
          }
        }
      }

      // l2 normalize the scores
      normalizeScores(tags);

      output_tags[sentence_index].push_back(tags);

      token_index++;
    }
  }

  // apply processing for model predictions
  for (size_t sentence_index = 0; sentence_index < output_tags.size();
       ++sentence_index) {
    auto cleaned_tokens =
        data::ner::utils::cleanAndLowerCase(tokens[sentence_index]);
    for (const auto& learned_tag : _tag_tracker->modelTags()) {
      learned_tag->processTags(output_tags[sentence_index], cleaned_tokens);
    }
  }

  return {output_tags, token_offsets};
}

data::LoaderPtr NerModel::getDataLoader(const dataset::DataSourcePtr& data,
                                        size_t batch_size, bool shuffle) const {
  auto csv_iter = data::CsvIterator::make(data, ',', 1000);
  return data::Loader::make(csv_iter, _supervised_transform, nullptr,
                            _bolt_inputs, {data::OutputColumns(_tags_column)},
                            /* batch_size= */ batch_size,
                            /* shuffle= */ shuffle, /* verbose= */ true,
                            /* shuffle_buffer_size= */ 20000);
}

ar::ConstArchivePtr NerModel::toArchive(bool with_optimizer) const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("model", _model->toArchive(with_optimizer));

  map->set("supervised_transform", _supervised_transform->toArchive());
  map->set("inference_transform", _inference_transform->toArchive());

  map->set("bolt_inputs", data::outputColumnsToArchive(_bolt_inputs));

  map->set("tokens_column", ar::str(_tokens_column));
  map->set("tags_column", ar::str(_tags_column));

  map->set("tag_tracker", _tag_tracker->toArchive());

  if (_rule) {
    map->set("use_rules_for", ar::vecStr(_rule->entities()));
  }

  return map;
}

std::unique_ptr<NerModel> NerModel::fromArchive(const ar::Archive& archive) {
  return std::make_unique<NerModel>(archive);
}

NerModel::NerModel(const ar::Archive& archive)
    : _model(bolt::Model::fromArchive(*archive.get("model"))),
      _supervised_transform(data::Transformation::fromArchive(
          *archive.get("supervised_transform"))),
      _inference_transform(data::Transformation::fromArchive(
          *archive.get("inference_transform"))),
      _bolt_inputs(data::outputColumnsFromArchive(*archive.get("bolt_inputs"))),
      _tokens_column(archive.str("tokens_column")),
      _tags_column(archive.str("tags_column")),
      _tag_tracker(std::make_unique<data::ner::utils::TagTracker>(
          *archive.get("tag_tracker"))) {
  if (archive.contains("use_rules_for")) {
    _rule = data::ner::getRuleForEntities(
        archive.getAs<ar::VecStr>("use_rules_for"));
  }

  auto ner_transformation_supervised = extractNerTokenizerTransform(
      _supervised_transform, /*is_inference=*/false);
  if (ner_transformation_supervised) {
    ner_transformation_supervised->setTagTracker(_tag_tracker);
  } else {
    throw std::logic_error("could not extract the supervised transform");
  }

  auto ner_transformation_inference =
      extractNerTokenizerTransform(_inference_transform, /*is_inference=*/true);
  if (ner_transformation_inference) {
    ner_transformation_inference->setTagTracker(_tag_tracker);
  } else {
    throw std::logic_error("could not extract the inference transform");
  }
}

}  // namespace thirdai::automl::udt
