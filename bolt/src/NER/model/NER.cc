#include "NER.h"
#include <bolt/src/NER/model/NerBoltModel.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace thirdai::bolt::NER {

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
  return _ner_backend_model->getTags(tokens, top_k);
}

ar::ConstArchivePtr NER::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(_ner_backend_model->type()));

  map->set("ner_backend_model", _ner_backend_model->toArchive());

  return map;
}

std::shared_ptr<NER> NER::fromArchive(const ar::Archive& archive) {
  std::string type = archive.getAs<std::string>("type");

  if (type == "bolt_ner") {
    std::shared_ptr<NerModelInterface> ner_backend_model =
        NerBoltModel::fromArchive(*archive.get("ner_backend_model"));
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

}  // namespace thirdai::bolt::NER