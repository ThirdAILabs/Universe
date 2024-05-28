#include "UDTNer.h"
#include <bolt/src/NER/model/NerUDTModel.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <archive/src/Archive.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/udt/Defaults.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/ner/NerTokenizationUnigram.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <pybind11/stl.h>
#include <utils/text/StringManipulation.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::automl::udt {

bolt::ModelPtr buildModel(
    uint32_t input_dim, uint32_t emb_dim, uint32_t output_dim,
    std::optional<std::vector<std::vector<float>*>> pretrained_emb) {
  auto input = bolt::Input::make(input_dim);

  auto emb_op = bolt::Embedding::make(emb_dim, input_dim, "relu",
                                      /* bias= */ true);
  if (pretrained_emb) {
    emb_op->setEmbeddings(pretrained_emb.value()[0]->data());
    emb_op->setBiases(pretrained_emb.value()[1]->data());
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

// data::TransformationPtr makeTransformation(
//     bool inference, const std::string& tags_column,
//     const std::string& tokens_column, const std::vector<std::string>& tags,
//     size_t input_dim, uint32_t dyadic_num_intervals,
//     const std::vector<dataset::TextTokenizerPtr>& target_word_tokenizers,
//     const std::optional<data::FeatureEnhancementConfig>& feature_config) {
//   std::optional<std::string> target_column = tags_column;
//   std::optional<size_t> target_dim = tags.size();
//   if (inference) {
//     target_column = std::nullopt;
//     target_dim = std::nullopt;
//   }

//   std::unordered_map<std::string, uint32_t> tag_to_label;
//   for (size_t i = 0; i < tags.size(); i++) {
//     tag_to_label[tags[i]] = i;
//   }

//   auto transform =
//       data::Pipeline::make({std::make_shared<data::NerTokenizerUnigram>(
//           /*tokens_column=*/tokens_column,
//           /*featurized_sentence_column=*/NER_FEATURIZED_SENTENCE,
//           /*target_column=*/target_column,
//           /*target_dim=*/target_dim,
//           /*dyadic_num_intervals=*/dyadic_num_intervals,
//           /*target_word_tokenizers=*/target_word_tokenizers,
//           /*feature_enhancement_config=*/feature_config,
//           /*tag_to_label=*/tag_to_label)});
//   transform = transform->then(std::make_shared<data::TextTokenizer>(
//       /*input_column=*/NER_FEATURIZED_SENTENCE,
//       /*output_indices=*/NER_FEATURIZED_SENTENCE,
//       /*output_values=*/std::nullopt,
//       /*tokenizer=*/
//       std::make_shared<dataset::NaiveSplitTokenizer>(
//           dataset::NaiveSplitTokenizer()),
//       /*encoder=*/
//       std::make_shared<dataset::NGramEncoder>(dataset::NGramEncoder(1)),
//       false, input_dim));
//   return transform;
// }

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

UDTNer::UDTNer(const ColumnDataTypes& data_types,
               const TokenTagsDataTypePtr& target,
               const std::string& target_name, const config::ArgumentMap& args)
    : _bolt_inputs({data::OutputColumns(NER_FEATURIZED_SENTENCE)}),
      _tokens_column(tokensColumn(data_types, target_name)),
      _tags_column(target_name),
      _label_to_tag(target->tags) {
  uint32_t input_dim =
      args.get<uint32_t>("input_dim", "integer", defaults::FEATURE_HASH_RANGE);
  if (args.contains("fhr")) {
    input_dim = args.get<uint32_t>("fhr", "integer");
  }

  uint32_t emb_dim = args.get<uint32_t>("embedding_dimension", "integer",
                                        defaults::NER_EMB_DIM);

  _model = buildModel(input_dim, emb_dim, target->tags.size(), std::nullopt);

  std::unordered_map<std::string, uint32_t> tag_to_label;
  for (size_t i = 0; i < target->tags.size(); i++) {
    tag_to_label[target->tags[i]] = i;
  }

  bolt::NER::NerUDTModel old(_tokens_column, _tags_column, tag_to_label,
                             target->target_tokenizers, target->feature_config);

  _supervised_transform = old._classifier->_train_transforms;
  _inference_transform = old._classifier->_inference_transforms;

  _bolt_inputs = old._classifier->_bolt_inputs;
  // _supervised_transform = makeTransformation(
  //     /*inference=*/false, /*tags_column=*/_tags_column,
  //     /*tokens_column=*/_tokens_column, _label_to_tag,
  //     /*input_dim=*/input_dim,
  //     /*dyadic_num_intervals=*/defaults::NER_DYADIC_INTERVALS,
  //     /*target_word_tokenizers=*/target->target_tokenizers,
  //     /*feature_config=*/target->feature_config);

  // _inference_transform = makeTransformation(
  //     /*inference=*/true, /*tags_column=*/_tags_column,
  //     /*tokens_column=*/_tokens_column, _label_to_tag,
  //     /*input_dim=*/input_dim,
  //     /*dyadic_num_intervals=*/defaults::NER_DYADIC_INTERVALS,
  //     /*target_word_tokenizers=*/target->target_tokenizers,
  //     /*feature_config=*/target->feature_config);
}

py::object UDTNer::train(const dataset::DataSourcePtr& data,
                         float learning_rate, uint32_t epochs,
                         const std::vector<std::string>& train_metrics,
                         const dataset::DataSourcePtr& val_data,
                         const std::vector<std::string>& val_metrics,
                         const std::vector<CallbackPtr>& callbacks,
                         TrainOptions options,
                         const bolt::DistributedCommPtr& comm,
                         py::kwargs kwargs) {
  (void)kwargs;

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

  return py::cast(history);
}

py::object UDTNer::evaluate(const dataset::DataSourcePtr& data,
                            const std::vector<std::string>& metrics,
                            bool sparse_inference, bool verbose,
                            py::kwargs kwargs) {
  (void)kwargs;

  bolt::Trainer trainer(_model);

  auto history = trainer.validate_with_data_loader(
      getDataLoader(data, defaults::BATCH_SIZE, /*shuffle=*/false),
      bolt::metrics::fromMetricNames(_model, metrics, "val_"), sparse_inference,
      verbose);

  return py::cast(history);
}

py::object UDTNer::predict(const MapInput& sample, bool sparse_inference,
                           bool return_predicted_class,
                           std::optional<uint32_t> top_k) {
  (void)return_predicted_class;

  if (!sample.count(_tokens_column)) {
    throw std::invalid_argument("Expected input to contain column '" +
                                _tokens_column + "'.");
  }

  auto tags = predictTags({sample.at(_tokens_column)}, sparse_inference,
                          top_k.value_or(1));

  return py::cast(tags[0]);
}

py::object UDTNer::predictBatch(const MapInputBatch& samples,
                                bool sparse_inference,
                                bool return_predicted_class,
                                std::optional<uint32_t> top_k) {
  (void)return_predicted_class;

  std::vector<std::string> sentences(samples.size());
  for (const auto& sample : samples) {
    if (!sample.count(_tokens_column)) {
      throw std::invalid_argument("Expected input to contain column '" +
                                  _tokens_column + "'.");
    }

    sentences.push_back(sample.at(_tokens_column));
  }

  auto tags = predictTags(sentences, sparse_inference, top_k.value_or(1));

  return py::cast(tags);
}

std::vector<SentenceTags> UDTNer::predictTags(
    const std::vector<std::string>& sentences, bool sparse_inference,
    uint32_t top_k) {
  std::vector<std::vector<std::string>> tokens;
  tokens.reserve(sentences.size());

  for (const auto& phrase : sentences) {
    tokens.push_back(text::split(phrase, ' '));
  }

  auto sentence_tokens =
      data::ArrayColumn<std::string>::make(std::move(tokens));
  auto data = data::ColumnMap({{_tokens_column, sentence_tokens}});

  auto featurized = _inference_transform->applyStateless(data);
  auto tensors =
      data::toTensorBatches(featurized, _bolt_inputs, defaults::BATCH_SIZE);

  size_t sentence_index = 0;
  size_t token_index = 0;

  std::vector<SentenceTags> output_tags(sentences.size());

  for (const auto& batch : tensors) {
    auto scores = _model->forward(batch, sparse_inference).at(0);

    for (size_t i = 0; i < scores->batchSize(); i++) {
      while (token_index == sentence_tokens->row(sentence_index).size()) {
        sentence_index++;
        token_index = 0;
      }

      auto top_labels = scores->getVector(i).topKNeurons(top_k);

      TokenTags tags;
      while (!top_labels.empty()) {
        float score = top_labels.top().first;
        auto tag = _label_to_tag.at(top_labels.top().second);
        top_labels.pop();
        tags.emplace_back(tag, score);
      }
      std::reverse(tags.begin(), tags.end());

      output_tags[sentence_index].push_back(tags);

      token_index++;
    }
  }

  return output_tags;
}

data::LoaderPtr UDTNer::getDataLoader(const dataset::DataSourcePtr& data,
                                      size_t batch_size, bool shuffle) const {
  auto data_iter =
      data::JsonIterator::make(data, {_tokens_column, _tags_column}, 1000);
  return data::Loader::make(data_iter, _supervised_transform, nullptr,
                            _bolt_inputs, {data::OutputColumns(_tags_column)},
                            /* batch_size= */ batch_size,
                            /* shuffle= */ shuffle, /* verbose= */ true,
                            /* shuffle_buffer_size= */ 20000);
}

ar::ConstArchivePtr UDTNer::toArchive(bool with_optimizer) const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("model", _model->toArchive(with_optimizer));

  map->set("supervised_transform", _supervised_transform->toArchive());
  map->set("inference_transform", _inference_transform->toArchive());

  map->set("bolt_inputs", data::outputColumnsToArchive(_bolt_inputs));

  map->set("tokens_column", ar::str(_tokens_column));
  map->set("tags_column", ar::str(_tags_column));

  map->set("label_to_tag", ar::vecStr(_label_to_tag));

  return map;
}

std::unique_ptr<UDTNer> UDTNer::fromArchive(const ar::Archive& archive) {
  return std::make_unique<UDTNer>(archive);
}

UDTNer::UDTNer(const ar::Archive& archive)
    : _model(bolt::Model::fromArchive(*archive.get("model"))),
      _supervised_transform(data::Transformation::fromArchive(
          *archive.get("supervised_transform"))),
      _inference_transform(data::Transformation::fromArchive(
          *archive.get("inference_transform"))),
      _bolt_inputs(data::outputColumnsFromArchive(*archive.get("bolt_inputs"))),
      _tokens_column(archive.str("tokens_column")),
      _tags_column(archive.str("tags_column")),
      _label_to_tag(archive.getAs<ar::VecStr>("label_to_tag")) {}

}  // namespace thirdai::automl::udt