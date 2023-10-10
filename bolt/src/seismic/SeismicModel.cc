#include "SeismicModel.h"
#include <cereal/archives/binary.hpp>
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/PatchEmbedding.h>
#include <bolt/src/nn/ops/PatchSum.h>
#include <bolt/src/seismic/SeismicLabels.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt::seismic {

SeismicModel::SeismicModel(size_t subcube_shape, size_t patch_shape,
                           size_t embedding_dim)
    : _subcube_shape(subcube_shape), _patch_shape(patch_shape) {
  std::tie(_model, _emb) =
      buildModel(nPatches(), patchDim(), embedding_dim, _n_output_classes);
}

void SeismicModel::train(const NumpyArray& subcubes,
                         const std::vector<SubcubeMetadata>& subcube_metadata,
                         float learning_rate, size_t batch_size) {
  if (static_cast<size_t>(subcubes.shape(0)) != subcube_metadata.size()) {
    throw std::invalid_argument(
        "Expected number of subcubes to match the number of subcube "
        "metadatas.");
  }
  auto data = convertToBatches(subcubes, batch_size);
  auto labels = makeLabelBatches(subcube_metadata, batch_size);

  bolt::LabeledDataset dataset =
      std::make_pair(std::move(data), std::move(labels));

  bolt::Trainer trainer(_model);

  // TODO(Nicholas) should we report loss?
  trainer.train(dataset, learning_rate, /* epochs= */ 1);
}

NumpyArray SeismicModel::embeddings(const NumpyArray& subcubes) {
  auto batch = convertToBatches(subcubes, subcubes.shape(0)).at(0);

  _model->forward(batch);

  return bolt::python::tensorToNumpy(_emb->tensor(),
                                     /* single_row_to_vector= */ false);
}

bolt::Dataset SeismicModel::convertToBatches(const NumpyArray& array,
                                             size_t batch_size) const {
  size_t n_patches = nPatches();
  size_t patch_dim = patchDim();
  if (array.ndim() != 3 || static_cast<size_t>(array.shape(1)) != n_patches ||
      static_cast<size_t>(array.shape(2)) != patch_dim) {
    throw std::invalid_argument(
        "Expected 3D numpy array of shape (n_subcubes, n_patches, "
        "patch_size).");
  }

  size_t stride_1 = array.strides(1);
  size_t stride_2 = array.strides(2);
  if (stride_1 != array.shape(2) * sizeof(float) || stride_2 != sizeof(float)) {
    throw std::invalid_argument("Expected array to be c_style and contiguous.");
  }

  size_t n_batches = (array.shape(0) + batch_size - 1) / batch_size;

  bolt::Dataset batches;
  for (size_t batch = 0; batch < n_batches; batch++) {
    size_t start = batch * batch_size;
    size_t end = std::min<size_t>(start + batch_size, array.shape(0));

    auto tensor =
        bolt::Tensor::fromArray(nullptr, array.data(start), end - start,
                                n_patches * patch_dim, n_patches * patch_dim,
                                /* with_grad= */ false);

    batches.push_back({tensor});
  }

  return batches;
}

bolt::Dataset SeismicModel::makeLabelBatches(
    const std::vector<SubcubeMetadata>& subcube_metadata,
    size_t batch_size) const {
  size_t n_batches = (subcube_metadata.size() + batch_size - 1) / batch_size;

  bolt::Dataset label_batches;

  for (size_t batch = 0; batch < n_batches; batch++) {
    size_t start = batch * batch_size;
    size_t end = std::min(start + batch_size, subcube_metadata.size());

    std::vector<uint32_t> indices;
    std::vector<float> values;
    std::vector<size_t> lens;

    for (size_t i = start; i < end; i++) {
      auto labels = seismicLabels(subcube_metadata[i], _subcube_shape,
                                  _label_cube_dim, _n_output_classes);

      indices.insert(indices.end(), labels.begin(), labels.end());

      float label_val = 1.0 / labels.size();
      for (size_t j = 0; j < labels.size(); j++) {
        values.push_back(label_val);
      }

      lens.push_back(labels.size());
    }

    auto tensor = bolt::Tensor::sparse(std::move(indices), std::move(values),
                                       std::move(lens), _n_output_classes);

    label_batches.push_back({tensor});
  }

  return label_batches;
}

std::pair<bolt::ModelPtr, bolt::ComputationPtr> SeismicModel::buildModel(
    size_t n_patches, size_t patch_dim, size_t embedding_dim,
    size_t n_output_classes) {
  auto patches = bolt::Input::make(n_patches * patch_dim);

  auto patch_emb =
      bolt::PatchEmbedding::make(
          /*emb_dim=*/100000, /*patch_dim=*/patch_dim,
          /*n_patches=*/n_patches, /*sparsity=*/0.01, /*activation=*/"relu")
          ->apply(patches);

  auto aggregated_embs = bolt::PatchSum::make(/*n_patches=*/n_patches,
                                              /*patch_dim=*/patch_emb->dim())
                             ->apply(patch_emb);

  auto emb = bolt::FullyConnected::make(
                 /*dim=*/embedding_dim, /*input_dim=*/aggregated_embs->dim(),
                 /* sparsity=*/1.0, /*activation=*/"tanh")
                 ->apply(aggregated_embs);

  auto output =
      bolt::FullyConnected::make(
          /*dim=*/n_output_classes, /*input_dim=*/emb->dim(), /*sparsity=*/0.05,
          /*activation=*/"softmax")
          ->apply(emb);

  auto loss = bolt::CategoricalCrossEntropy::make(
      output, bolt::Input::make(n_output_classes));

  auto model = bolt::Model::make({patches}, {output}, {loss});

  return {model, emb};
}

void SeismicModel::save(const std::string& filename) const {
  auto output = dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(output);
}

void SeismicModel::save_stream(std::ostream& output) const {
  cereal::BinaryOutputArchive oarchive(output);
  oarchive(*this);
}

std::shared_ptr<SeismicModel> SeismicModel::load(const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(input_stream);
}

std::shared_ptr<SeismicModel> SeismicModel::load_stream(std::istream& input) {
  cereal::BinaryInputArchive iarchive(input);
  std::shared_ptr<SeismicModel> deserialize_into(new SeismicModel());
  iarchive(*deserialize_into);

  return deserialize_into;
}

template <class Archive>
void SeismicModel::serialize(Archive& archive) {
  archive(_model, _emb, _subcube_shape, _patch_shape, _label_cube_dim,
          _n_output_classes);
}

}  // namespace thirdai::bolt::seismic