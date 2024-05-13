#include "NER.h"
#include <bolt/src/NER/model/NerPretrainedModel.h>
#include <bolt/src/NER/model/NerUDTModel.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace thirdai::bolt {

// need to use label tag as transform for label during training
metrics::History NER::train(const dataset::DataSourcePtr& train_data,
                            float learning_rate, uint32_t epochs,
                            size_t batch_size,
                            const std::vector<std::string>& train_metrics,
                            const dataset::DataSourcePtr& val_data,
                            const std::vector<std::string>& val_metrics) {
  return _ner_backend_model->train(train_data, learning_rate, epochs,
                                   batch_size, train_metrics, val_data,
                                   val_metrics);
}

std::vector<std::vector<std::vector<std::pair<std::string, float>>>>
NER::getNerTags(std::vector<std::vector<std::string>>& tokens, uint32_t top_k) {
  std::vector<PerTokenListPredictions> tags_and_scores =
      _ner_backend_model->getTags(tokens, top_k);

  std::vector<std::vector<std::vector<std::pair<std::string, float>>>>
      string_and_scores;

  for (const auto& sentence_tags_and_scores : tags_and_scores) {
    std::vector<std::vector<std::pair<std::string, float>>>
        sentence_string_tags_and_scores;
    sentence_string_tags_and_scores.reserve(sentence_tags_and_scores.size());
    for (const auto& tags_and_scores : sentence_tags_and_scores) {
      std::vector<std::pair<std::string, float>> token_tags_and_scores;
      token_tags_and_scores.reserve(tags_and_scores.size());
      for (const auto& tag_and_score : tags_and_scores) {
        token_tags_and_scores.push_back(
            {_label_to_tag_map[tag_and_score.first], tag_and_score.second});
      }
      sentence_string_tags_and_scores.push_back(token_tags_and_scores);
    }
    string_and_scores.push_back(sentence_string_tags_and_scores);
    ;
  }
  return string_and_scores;
}

ar::ConstArchivePtr NER::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(_ner_backend_model->type()));

  map->set("ner_backend_model", _ner_backend_model->toArchive());

  return map;
}

std::shared_ptr<NER> NER::fromArchive(const ar::Archive& archive) {
  std::string type = archive.getAs<std::string>("type");

  if (type == "pretrained_ner") {
    std::shared_ptr<bolt::NerBackend> ner_backend_model =
        bolt::NerPretrainedModel::fromArchive(
            *archive.get("ner_backend_model"));
    return std::make_shared<NER>(NER(ner_backend_model));
  }
  if (type == "ner") {
    std::shared_ptr<bolt::NerBackend> ner_backend_model =
        bolt::NerUDTModel::fromArchive(*archive.get("ner_backend_model"));
    return std::make_shared<NER>(NER(ner_backend_model));
  }
  throw std::invalid_argument("Cannot load a NER backend of type: " + type);
}

void NER::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void NER::save_stream(std::ostream& output) const {
  ar::serialize(toArchive(), output);
}

std::shared_ptr<NER> NER::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

std::shared_ptr<NER> NER::load_stream(std::istream& input) {
  auto archive = ar::deserialize(input);
  return fromArchive(*archive);
}

}  // namespace thirdai::bolt