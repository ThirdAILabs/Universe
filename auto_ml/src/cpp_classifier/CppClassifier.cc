#include "CppClassifier.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <auto_ml/src/featurization/Featurizer.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <memory>
#include <stdexcept>

namespace thirdai {

CppClassifier::CppClassifier(std::shared_ptr<automl::Featurizer> featurizer,
                             std::shared_ptr<bolt::Model> model)
    : _featurizer(std::move(featurizer)), _model(std::move(model)) {
  auto comps = _model->computationOrder();
  bool ops_compatible =
      comps.size() == 3 &&
      std::dynamic_pointer_cast<bolt::Input>(comps[0]->op()) &&
      bolt::Embedding::cast(comps[1]->op()) &&
      bolt::FullyConnected::cast(comps[2]->op());

  bool loss_compatible =
      _model->losses().size() == 1 &&
      std::dynamic_pointer_cast<bolt::CategoricalCrossEntropy>(
          _model->losses()[0]);

  if (!ops_compatible || !loss_compatible) {
    throw std::invalid_argument(
        "Model architecture is not compatible for use with CppClassifier.");
  }
}

CppClassifier::CppClassifier() {}

std::shared_ptr<CppClassifier> CppClassifier::load(
    const std::string& saved_model) {
  auto istream = dataset::SafeFileIO::ifstream(saved_model);
  cereal::BinaryInputArchive iarchive(istream);

  auto deserialize_into = std::make_shared<CppClassifier>();

  iarchive(*deserialize_into);

  return deserialize_into;
}

uint32_t CppClassifier::predict(
    const std::unordered_map<std::string, std::string>& input) {
  auto output = _model->forward(_featurizer->featurizeInput(input)).at(0);

  return output->getVector(0).getHighestActivationId();
}

template void CppClassifier::serialize(cereal::BinaryInputArchive& archive);
template void CppClassifier::serialize(cereal::BinaryOutputArchive& archive);

template <class Archive>
void CppClassifier::serialize(Archive& archive) {
  archive(_featurizer, _model);
}

}  // namespace thirdai