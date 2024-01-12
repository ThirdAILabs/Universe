#include "CppClassifier.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/featurization/Featurizer.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <memory>

namespace thirdai {

std::shared_ptr<CppClassifier> CppClassifier::load(
    const std::string& saved_model) {
  auto istream = dataset::SafeFileIO::ifstream(saved_model);
  cereal::BinaryInputArchive iarchive(istream);

  std::shared_ptr<CppClassifier> deserialize_into(new CppClassifier());

  iarchive(*deserialize_into);

  return deserialize_into;
}

uint32_t CppClassifier::predict(
    const std::unordered_map<std::string, std::string>& input) {
  auto output = _model->forward(_featurizer->featurizeInput(input)).at(0);

  return output->getVector(0).getHighestActivationId();
}

template <class Archive>
void CppClassifier::serialize(Archive& archive) {
  // archive(_featurizer, _model);
  (void)archive;
}

}  // namespace thirdai