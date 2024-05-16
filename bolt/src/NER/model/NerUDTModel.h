#pragma once

#include "NerClassifier.h"
#include <bolt/src/NER/Defaults.h>
#include <bolt/src/NER/model/NerBackend.h>
#include <bolt/src/nn/model/Model.h>
#include <data/src/transformations/Pipeline.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <memory>

namespace thirdai::bolt::NER {

class NerUDTModel;

class NerUDTModel final : public NerModelInterface {
 public:
  std::string type() const final { return "udt_ner"; }
  NerUDTModel(bolt::ModelPtr model, std::string tokens_column,
              std::string tags_column,
              std::unordered_map<std::string, uint32_t> tag_to_label,
              std::vector<dataset::TextTokenizerPtr> target_word_tokenizers);

  NerUDTModel(std::string tokens_column, std::string tags_column,
              std::unordered_map<std::string, uint32_t> tag_to_label,
              std::vector<dataset::TextTokenizerPtr> target_word_tokenizers);

  NerUDTModel(std::shared_ptr<NerUDTModel>& pretrained_model,
              std::string tokens_column, std::string tags_column,
              std::unordered_map<std::string, uint32_t> tag_to_label);

  static bolt::ModelPtr initializeBoltModel(
      uint32_t input_dim, uint32_t emb_dim, uint32_t output_dim,
      std::optional<std::vector<std::vector<float>*>> pretrained_emb =
          std::nullopt);

  std::vector<PerTokenListPredictions> getTags(
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

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<NerUDTModel> load(const std::string& filename);

  static std::shared_ptr<NerUDTModel> load_stream(std::istream& input_stream);

  bolt::ModelPtr getBoltModel() final { return _bolt_model; }

  std::vector<dataset::TextTokenizerPtr> getTargetWordTokenizers() {
    return _target_word_tokenizers;
  }

  std::string getTokensColumn() const final { return _tokens_column; }

  std::string getTagsColumn() const final { return _tags_column; }

  NerUDTModel() = default;
  ~NerUDTModel() override = default;

 private:
  void initializeNER(uint32_t fhr, uint32_t number_labels);

  bolt::ModelPtr _bolt_model;
  std::string _tokens_column, _tags_column;
  std::vector<dataset::TextTokenizerPtr> _target_word_tokenizers;

  std::unordered_map<std::string, uint32_t> _tag_to_label;

  uint32_t _dyadic_num_intervals = defaults::UDT_DYADIC_NUM_INTERVALS;

  std::string _featurized_sentence_column =
      "featurized_sentence_for_" + _tokens_column;

  NerClassifierPtr _classifier;
};
}  // namespace thirdai::bolt::NER