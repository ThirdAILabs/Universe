#include "NerModel.h"
#include <bolt/src/NER/model/NerBoltModel.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <cmath>
#include <memory>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

namespace thirdai::bolt {

// need to use label tag as transform for label during training
metrics::History NerModel::train(const dataset::DataSourcePtr& train_data,
                                 float learning_rate, uint32_t epochs,
                                 size_t batch_size,
                                 const std::vector<std::string>& train_metrics,
                                 const dataset::DataSourcePtr& val_data,
                                 const std::vector<std::string>& val_metrics) {
  return _ner_backend_model->train(train_data, learning_rate, epochs,
                                   batch_size, train_metrics, val_data,
                                   val_metrics);
}

std::vector<std::vector<std::string>> NerModel::getNerTags(
    std::vector<std::vector<std::string>>& tokens) {
  std::vector<std::vector<uint32_t>> tags = _ner_backend_model->getTags(tokens);

  std::vector<std::vector<std::string>> string_tags(tags.size());
  for (const auto& sentence_tag : tags) {
    std::vector<std::string> sentence_string_tags;
    sentence_string_tags.reserve(sentence_tag.size());
    for (const auto& tag : sentence_tag) {
      sentence_string_tags.push_back(_tag_to_label_map[tag]);
    }
  }
  return string_tags;
}

ar::ConstArchivePtr NerModel::toArchive() const {
  auto ner_bolt_model = ar::Map::make();

  ner_bolt_model->set("ner_backend_model", _ner_backend_model->toArchive());

  ar::MapStrU64 label_to_tags;
  for (const auto& [label, tag] : _label_to_tag_map) {
    label_to_tags[label] = tag;
  }
  ner_bolt_model->set("label_to_tag_map", ar::mapStrU64(label_to_tags));

  return ner_bolt_model;
}

std::shared_ptr<NerModel> NerModel::fromArchive(const ar::Archive& archive) {
  std::shared_ptr<bolt::NerBackend> ner_backend_model =
      bolt::NerBoltModel::fromArchive(*archive.get("ner_backend_model"));
  std::unordered_map<std::string, uint32_t> label_to_tags;
  for (const auto& [k, v] : archive.getAs<ar::MapStrU64>("string_to_id")) {
    label_to_tags[k] = v;
  }
  return std::make_shared<NerModel>(NerModel(ner_backend_model, label_to_tags));
}

void NerModel::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void NerModel::save_stream(std::ostream& output) const {
  ar::serialize(toArchive(), output);
}

std::shared_ptr<NerModel> NerModel::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

std::shared_ptr<NerModel> NerModel::load_stream(std::istream& input) {
  auto archive = ar::deserialize(input);
  return fromArchive(*archive);
}

}  // namespace thirdai::bolt