#pragma once

#include <bolt/src/NER/model/NerModel.h>
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

// added to support both bolt and udt back in case
class NerBackend {
 public:
  virtual ~NerBackend() = default;
  // initialize
  virtual std::vector<std::vector<uint32_t>> getTags(
      std::vector<std::vector<std::string>> tokens) = 0;

  virtual metrics::History train(
      const dataset::DataSourcePtr& train_data, float learning_rate,
      uint32_t epochs, size_t batch_size,
      const std::vector<std::string>& train_metrics,
      const dataset::DataSourcePtr& val_data,
      const std::vector<std::string>& val_metrics) = 0;

  virtual ar::ConstArchivePtr toArchive() const = 0;
};

class NerModel;

class NerModel : public std::enable_shared_from_this<NerModel> {

 public:
  NerModel(std::shared_ptr<NerBackend> model,
           std::unordered_map<std::string, uint32_t> label_to_tag_map)
      : _ner_backend_model(std::move(model)),
        _label_to_tag_map(std::move(label_to_tag_map)) {
    for (const auto& [k, v] : _label_to_tag_map) {
      _tag_to_label_map[v] = k;
    }
  }

  // initialize (flag)
  metrics::History train(const dataset::DataSourcePtr& train_data,
                         float learning_rate, uint32_t epochs,
                         size_t batch_size,
                         const std::vector<std::string>& train_metrics,
                         const dataset::DataSourcePtr& val_data,
                         const std::vector<std::string>& val_metrics);

  std::vector<std::vector<std::string>> getNerTags(
      std::vector<std::vector<std::string>>& tokens);

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<NerModel> fromArchive(const ar::Archive& archive);

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<NerModel> load(const std::string& filename);

  static std::shared_ptr<NerModel> load_stream(std::istream& input_stream);

 private:
  std::shared_ptr<NerBackend> _ner_backend_model;

  std::unordered_map<std::string, uint32_t> _label_to_tag_map;
  std::unordered_map<uint32_t, std::string> _tag_to_label_map;

  NerModel() {}
};

}  // namespace thirdai::bolt