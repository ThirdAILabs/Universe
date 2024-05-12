#pragma once

#include <bolt/src/NER/model/NER.h>
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
#include <unordered_set>
#include <utility>

namespace thirdai::bolt {

using PerTokenPredictions = std::vector<std::pair<uint32_t, float>>;
using PerTokenListPredictions = std::vector<PerTokenPredictions>;

// added to support both bolt and udt back in case
class NerBackend {
 public:
  virtual ~NerBackend() = default;
  // initialize
  virtual std::vector<PerTokenListPredictions> getTags(
      std::vector<std::vector<std::string>> tokens, uint32_t top_k) = 0;

  virtual metrics::History train(
      const dataset::DataSourcePtr& train_data, float learning_rate,
      uint32_t epochs, size_t batch_size,
      const std::vector<std::string>& train_metrics,
      const dataset::DataSourcePtr& val_data,
      const std::vector<std::string>& val_metrics) = 0;

  virtual ar::ConstArchivePtr toArchive() const = 0;

  virtual std::string type() const = 0;

  virtual std::unordered_map<std::string, uint32_t> getTagToLabel() = 0;
};

class NER;

class NER : public std::enable_shared_from_this<NER> {
 public:
  explicit NER(std::shared_ptr<NerBackend> model)
      : _ner_backend_model(std::move(model)) {
    _tag_to_label_map = _ner_backend_model->getTagToLabel();
    for (const auto& [k, v] : _tag_to_label_map) {
      _label_to_tag_map[v] = k;
    }
  }

  // initialize (flag)
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
  std::shared_ptr<NerBackend> _ner_backend_model;

  std::unordered_map<std::string, uint32_t> _tag_to_label_map;
  std::unordered_map<uint32_t, std::string> _label_to_tag_map;

  NER() {}
};

}  // namespace thirdai::bolt