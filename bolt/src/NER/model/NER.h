#pragma once

#include <bolt/src/NER/model/NerBackend.h>
#include <bolt/src/NER/model/NerBoltModel.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <licensing/src/CheckLicense.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace thirdai::bolt::NER {

class NER : public std::enable_shared_from_this<NER> {
 public:
  explicit NER(std::shared_ptr<NerModelInterface> model)
      : _ner_backend_model(std::move(model)) {}

  NER(const std::string& model_path, std::string tokens_column,
      std::string tags_column,
      std::unordered_map<std::string, uint32_t> tag_to_label) {
    auto ner_model = load(model_path);
    auto ner_backend = ner_model->getBackend();
    if (ner_backend->type() == "bolt_ner") {
      auto ner_pretrained_model =
          std::dynamic_pointer_cast<NerBoltModel>(ner_backend);
      _ner_backend_model = std::make_shared<NerBoltModel>(
          ner_pretrained_model, std::move(tokens_column),
          std::move(tags_column), std::move(tag_to_label));
    } else {
      std::stringstream error;
      error << "The backend type: " << ner_backend->type()
            << " is not a supported type.";
      throw std::logic_error(error.str());
    }
  }

  bolt::ModelPtr getModel() { return _ner_backend_model->getBoltModel(); }

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

  std::shared_ptr<NerModelInterface> getBackend() { return _ner_backend_model; }

  std::string type() { return _ner_backend_model->type(); }

  std::string getTokensColumn() {
    return _ner_backend_model->getTokensColumn();
  }

  std::string getTagsColumn() { return _ner_backend_model->getTagsColumn(); }

 private:
  std::shared_ptr<NerModelInterface> _ner_backend_model;

  NER() {}
};

}  // namespace thirdai::bolt::NER