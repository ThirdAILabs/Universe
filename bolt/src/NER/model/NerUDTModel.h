#pragma once

#include "NerClassifier.h"
#include <bolt/src/NER/Defaults.h>
#include <bolt/src/NER/model/NerBackend.h>
#include <bolt/src/nn/model/Model.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/ner/NerTokenizationUnigram.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <memory>

namespace thirdai::bolt::NER {

class NerUDTModel final : public NerModelInterface {
 public:
  std::string type() const final { return "udt_ner"; }
  NerUDTModel(
      bolt::ModelPtr model, std::string tokens_column, std::string tags_column,
      std::unordered_map<std::string, uint32_t> tag_to_label,
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      std::optional<data::FeatureEnhancementConfig> feature_enhancement_config);

  NerUDTModel(
      std::string tokens_column, std::string tags_column,
      std::unordered_map<std::string, uint32_t> tag_to_label,
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      std::optional<data::FeatureEnhancementConfig> feature_enhancement_config);

  NerUDTModel(std::shared_ptr<NerUDTModel>& pretrained_model,
              std::string tokens_column, std::string tags_column,
              std::unordered_map<std::string, uint32_t> tag_to_label,
              const std::optional<data::FeatureEnhancementConfig>&
                  feature_enhancement_config);

  data::TransformationPtr getTransformations(bool inference, size_t fhr,
                                             size_t num_label) {
    std::optional<std::string> target_column =
        inference ? std::optional<std::string>{}
                  : std::optional<std::string>{_tags_column};
    std::optional<size_t> target_dim =
        inference ? std::optional<size_t>{} : std::optional<size_t>{num_label};

    auto transform = data::Pipeline::make(
        {std::make_shared<thirdai::data::NerTokenizerUnigram>(
            /*tokens_column=*/_tokens_column,
            /*featurized_sentence_column=*/_featurized_sentence_column,
            /*target_column=*/target_column,
            /*target_dim=*/target_dim,
            /*dyadic_num_intervals=*/_dyadic_num_intervals,
            /*target_word_tokenizers=*/_target_word_tokenizers,
            /*feature_enhancement_config=*/_feature_enhancement_config,
            /*tag_to_label=*/_tag_to_label)});
    transform = transform->then(std::make_shared<data::TextTokenizer>(
        /*input_column=*/_featurized_sentence_column,
        /*output_indices=*/_featurized_tokens_indices_column,
        /*output_values=*/std::nullopt,
        /*tokenizer=*/
        std::make_shared<dataset::NaiveSplitTokenizer>(
            dataset::NaiveSplitTokenizer()),
        /*encoder=*/
        std::make_shared<dataset::NGramEncoder>(dataset::NGramEncoder(1)),
        false, fhr));
    return transform;
  }

  static bolt::ModelPtr initializeBoltModel(
      uint32_t input_dim, uint32_t emb_dim, uint32_t output_dim,
      std::optional<std::vector<std::vector<float>*>> pretrained_emb =
          std::nullopt);

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

  static std::shared_ptr<NerUDTModel> fromArchive(const ar::Archive& archive);

  bolt::ModelPtr getBoltModel() final { return _bolt_model; }

  std::vector<dataset::TextTokenizerPtr> getTargetWordTokenizers() {
    return _target_word_tokenizers;
  }

  std::string getTokensColumn() const final { return _tokens_column; }

  std::string getTagsColumn() const final { return _tags_column; }

  NerUDTModel() = default;
  ~NerUDTModel() override = default;

//  private:
  void initializeNER(uint32_t fhr, uint32_t number_labels);

  bolt::ModelPtr _bolt_model;
  std::string _tokens_column, _tags_column;
  std::vector<dataset::TextTokenizerPtr> _target_word_tokenizers;

  std::unordered_map<std::string, uint32_t> _tag_to_label;
  std::unordered_map<uint32_t, std::string> _label_to_tag_map;

  std::optional<data::FeatureEnhancementConfig> _feature_enhancement_config;

  const std::string _featurized_tokens_indices_column =
      "featurized_tokens_indices_column";

  uint32_t _dyadic_num_intervals = defaults::UDT_DYADIC_NUM_INTERVALS;

  std::string _featurized_sentence_column;

  NerClassifierPtr _classifier;
};

}  // namespace thirdai::bolt::NER