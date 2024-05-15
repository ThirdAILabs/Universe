#pragma once

#include <bolt/src/NER/model/NerBackend.h>
#include <bolt/src/NER/model/NerPretrainedModel.h>
#include <bolt/src/NER/model/NerUDTModel.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <licensing/src/CheckLicense.h>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>

namespace thirdai::bolt {

class NER;

class NER : public std::enable_shared_from_this<NER> {
 public:
  explicit NER(const std::shared_ptr<NerModelInterface>& model)
      : _ner_backend_model(model) {
    auto tag_to_label_map = _ner_backend_model->getTagToLabel();
    for (const auto& [k, v] : tag_to_label_map) {
      _label_to_tag_map[v] = k;
    }
  }

  NER(std::string tokens_column, std::string tags_column,
      std::unordered_map<std::string, uint32_t> tag_to_label,
      std::optional<std::string> pretrained_model_path = std::nullopt,
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers =
          std::vector<dataset::TextTokenizerPtr>(
              {std::make_shared<dataset::NaiveSplitTokenizer>(),
               std::make_shared<dataset::CharKGramTokenizer>(4)})) {
    if (pretrained_model_path) {
      auto model = std::make_shared<NerPretrainedModel>(
          *pretrained_model_path, std::move(tokens_column),
          std::move(tags_column), std::move(tag_to_label));
      _ner_backend_model = std::static_pointer_cast<NerModelInterface>(model);

    } else {
      std::cout << "The total number of elements in the unordered_map is: "
                << tag_to_label.size() << std::endl;
      auto model = std::make_shared<NerUDTModel>(
          std::move(tokens_column), std::move(tags_column),
          std::move(tag_to_label), std::move(target_word_tokenizers));
      _ner_backend_model = std::static_pointer_cast<NerModelInterface>(model);
    }

    auto tag_to_label_map = _ner_backend_model->getTagToLabel();
    for (const auto& [k, v] : tag_to_label_map) {
      _label_to_tag_map[v] = k;
    }
  }

  metrics::History train(const dataset::DataSourcePtr& train_data,
                         float learning_rate, uint32_t epochs,
                         size_t batch_size,
                         const std::vector<std::string>& train_metrics,
                         const dataset::DataSourcePtr& val_data,
                         const std::vector<std::string>& val_metrics);

  std::vector<std::vector<std::vector<std::pair<std::string, float>>>>
  getNerTags(std::vector<std::vector<std::string>>& tokens, uint32_t top_k);

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<NER> fromArchive(const ar::Archive& archive);

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<NER> load(const std::string& filename);

  static std::shared_ptr<NER> load_stream(std::istream& input_stream);

 private:
  std::shared_ptr<NerModelInterface> _ner_backend_model;

  std::unordered_map<uint32_t, std::string> _label_to_tag_map;

  NER() {}
};

}  // namespace thirdai::bolt