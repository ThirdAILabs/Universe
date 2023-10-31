#include "SeismicClassifier.h"
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/PatchEmbedding.h>
#include <bolt/src/nn/ops/PatchSum.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <memory>

namespace thirdai::bolt::seismic {

SeismicClassifier::SeismicClassifier(
    const std::shared_ptr<SeismicBase>& emb_model, size_t n_classes,
    bool freeze_emb_model)
    : SeismicBase(
          /* input_shape_data= */ emb_model->inputShapeData(),
          /* model= */
          addClassifierHead(emb_model->getModel(), n_classes, freeze_emb_model),
          /* embedding_last= */ false) {}

metrics::History SeismicClassifier::trainOnPatches(
    const NumpyArray& subcubes, std::vector<std::vector<uint32_t>> labels,
    float learning_rate, size_t batch_size,
    const std::vector<callbacks::CallbackPtr>& callbacks,
    std::optional<uint32_t> log_interval, const DistributedCommPtr& comm) {
  if (static_cast<size_t>(subcubes.shape(0)) != labels.size()) {
    throw std::invalid_argument(
        "Expected number of subcubes to match the number of subcube "
        "metadatas.");
  }

  return SeismicBase::trainOnPatches(
      subcubes, makeLabelbatches(std::move(labels), batch_size), learning_rate,
      batch_size, callbacks, log_interval, comm);
}

NumpyArray SeismicClassifier::predictionsForPatches(
    const NumpyArray& subcubes) {
  auto batch = convertToBatches(subcubes, subcubes.shape(0)).at(0);

  auto output = _model->forward(batch).at(0);

  return python::tensorToNumpy(output, /* single_row_to_vector= */ false);
}

Dataset SeismicClassifier::makeLabelbatches(
    std::vector<std::vector<uint32_t>> labels, size_t batch_size) {
  data::ColumnMap column({{"col", data::ArrayColumn<uint32_t>::make(
                                      std::move(labels), labelDim())}});

  return data::toTensorBatches(
      column, {data::OutputColumns("col", data::ValueFillType::SumToOne)},
      batch_size);
}

float outputSparsity(size_t n_classes) {
  if (n_classes < 500) {
    return 1.0;
  }

  if (n_classes < 10000) {
    return 0.2;
  }

  return 2000.F / n_classes;
}

ModelPtr SeismicClassifier::addClassifierHead(const ModelPtr& emb_model,
                                              size_t n_classes,
                                              bool freeze_emb_model) {
  auto patches = Input::make(emb_model->inputDims().at(0));

  auto patch_emb_op =
      std::dynamic_pointer_cast<PatchEmbedding>(emb_model->getOp("patch_emb"));
  auto patch_emb = patch_emb_op->apply(patches);

  auto patch_sum_op =
      std::dynamic_pointer_cast<PatchSum>(emb_model->getOp("patch_sum"));
  auto patch_sum = patch_sum_op->apply(patch_emb);

  auto emb_op =
      std::dynamic_pointer_cast<FullyConnected>(emb_model->getOp("emb"));
  auto emb = emb_op->apply(patch_sum);

  if (freeze_emb_model) {
    patch_emb_op->setTrainable(false);
    patch_sum_op->setTrainable(false);
    emb_op->setTrainable(false);
  }

  auto output = FullyConnected::make(n_classes, emb->dim(),
                                     outputSparsity(n_classes), "softmax")
                    ->apply(emb);

  auto loss = CategoricalCrossEntropy::make(output, Input::make(n_classes));

  return Model::make({patches}, {output}, {loss});
}

void SeismicClassifier::save(const std::string& filename) const {
  auto output = dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(output);
}

void SeismicClassifier::save_stream(std::ostream& output) const {
  cereal::BinaryOutputArchive oarchive(output);
  getModel()->setSerializeOptimizer(/* should_save_optimizer= */ false);
  oarchive(*this);
}

std::shared_ptr<SeismicClassifier> SeismicClassifier::load(
    const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(input_stream);
}

std::shared_ptr<SeismicClassifier> SeismicClassifier::load_stream(
    std::istream& input) {
  cereal::BinaryInputArchive iarchive(input);
  std::shared_ptr<SeismicClassifier> deserialize_into(new SeismicClassifier());
  iarchive(*deserialize_into);

  return deserialize_into;
}

template void SeismicClassifier::serialize(cereal::BinaryInputArchive&);
template void SeismicClassifier::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void SeismicClassifier::serialize(Archive& archive) {
  archive(cereal::base_class<SeismicBase>(this));
}

}  // namespace thirdai::bolt::seismic

CEREAL_REGISTER_TYPE(thirdai::bolt::seismic::SeismicClassifier)