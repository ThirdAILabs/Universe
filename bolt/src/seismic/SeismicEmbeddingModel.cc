#include "SeismicEmbeddingModel.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/tuple.hpp>
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/PatchEmbedding.h>
#include <bolt/src/nn/ops/PatchSum.h>
#include <bolt/src/seismic/SeismicLabels.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt/src/utils/Timer.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt::seismic {

void checkNonzeroDims(const Shape& a) {
  auto [x, y, z] = a;
  if (x == 0 || y == 0 || z == 0) {
    throw std::invalid_argument(
        "Expected all dimensions of shape to be nonzero.");
  }
}

bool shapesAreMultiples(const Shape& a, const Shape& b) {
  auto [a_x, a_y, a_z] = a;
  auto [b_x, b_y, b_z] = b;
  return (a_x % b_x == 0) && (a_y % b_y == 0) && (a_z % b_z == 0);
}

SeismicEmbeddingModel::SeismicEmbeddingModel(size_t subcube_shape,
                                             size_t patch_shape,
                                             size_t embedding_dim,
                                             std::optional<size_t> max_pool)
    : _subcube_shape(subcube_shape, subcube_shape, subcube_shape),
      _patch_shape(patch_shape, patch_shape, patch_shape),
      _max_pool(max_pool
                    ? std::make_optional<Shape>(*max_pool, *max_pool, *max_pool)
                    : std::nullopt) {
  checkNonzeroDims(_subcube_shape);
  checkNonzeroDims(_patch_shape);
  if (_max_pool) {
    checkNonzeroDims(*_max_pool);
  }

  if (!shapesAreMultiples(_subcube_shape, _patch_shape)) {
    throw std::invalid_argument(
        "Subcube shape must be a multiple of the patch shape.");
  }
  if (_max_pool && !shapesAreMultiples(_patch_shape, *_max_pool)) {
    throw std::invalid_argument(
        "Max pool shape must be a multiple of the patch shape.");
  }
  std::tie(_model, _emb) = buildModel(nPatches(), flattenedPatchDim(),
                                      embedding_dim, _n_output_classes);

#if THIRDAI_EXPOSE_ALL
  _model->summary();
#endif
}

metrics::History SeismicEmbeddingModel::trainOnPatches(
    const NumpyArray& subcubes,
    const std::vector<SubcubeMetadata>& subcube_metadata, float learning_rate,
    size_t batch_size, const std::vector<callbacks::CallbackPtr>& callbacks,
    std::optional<uint32_t> log_interval, const DistributedCommPtr& comm) {
  if (static_cast<size_t>(subcubes.shape(0)) != subcube_metadata.size()) {
    throw std::invalid_argument(
        "Expected number of subcubes to match the number of subcube "
        "metadatas.");
  }

  utils::Timer timer;

  auto data = convertToBatches(subcubes, batch_size);
  auto labels = makeLabelBatches(subcube_metadata, batch_size);

  timer.stop();

  std::cout << "Created " << data.size() << " batches from "
            << subcubes.shape(0) << " subcubes in " << timer.seconds()
            << " seconds." << std::endl;

  LabeledDataset dataset = std::make_pair(std::move(data), std::move(labels));

  Trainer trainer(_model, std::nullopt, python::CtrlCCheck());

  if (comm) {
    _model->disableSparseParameterUpdates();
  }

  auto metrics = trainer.train_with_metric_names(
      dataset, learning_rate, /* epochs= */ 1,
      /* train_metrics= */ {"loss"}, /* validation_data= */ std::nullopt,
      /* validation_metrics= */ {}, /* steps_per_validation= */ std::nullopt,
      /* use_sparsity_in_validation= */ false, /* callbacks= */ callbacks,
      /* autotune_rehash_rebuild= */ false, /* verbose= */ true,
      /* logging_interval= */ log_interval, /* comm= */ comm);

  if (comm) {
    _model->enableSparseParameterUpdates();
  }

  return metrics;
}

NumpyArray SeismicEmbeddingModel::embeddingsForPatches(
    const NumpyArray& subcubes) {
  auto batch = convertToBatches(subcubes, subcubes.shape(0)).at(0);

  _model->forward(batch);

  return python::tensorToNumpy(_emb->tensor(),
                               /* single_row_to_vector= */ false);
}

Dataset SeismicEmbeddingModel::convertToBatches(const NumpyArray& array,
                                                size_t batch_size) const {
  size_t n_patches = nPatches();
  size_t flattened_patch_dim = flattenedPatchDim();
  if (array.ndim() != 3 || static_cast<size_t>(array.shape(1)) != n_patches ||
      static_cast<size_t>(array.shape(2)) != flattened_patch_dim) {
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

  Dataset batches;
  for (size_t batch = 0; batch < n_batches; batch++) {
    size_t start = batch * batch_size;
    size_t end = std::min<size_t>(start + batch_size, array.shape(0));

    auto tensor = Tensor::fromArray(nullptr, array.data(start), end - start,
                                    n_patches * flattened_patch_dim,
                                    n_patches * flattened_patch_dim,
                                    /* with_grad= */ false);

    batches.push_back({tensor});
  }

  return batches;
}

Dataset SeismicEmbeddingModel::makeLabelBatches(
    const std::vector<SubcubeMetadata>& subcube_metadata,
    size_t batch_size) const {
  size_t n_batches = (subcube_metadata.size() + batch_size - 1) / batch_size;

  Dataset label_batches;

  for (size_t batch = 0; batch < n_batches; batch++) {
    size_t start = batch * batch_size;
    size_t end = std::min(start + batch_size, subcube_metadata.size());

    std::vector<uint32_t> indices;
    std::vector<float> values;
    std::vector<size_t> lens;

    for (size_t i = start; i < end; i++) {
      auto labels =
          seismicLabelsFromMetadata(subcube_metadata[i], _subcube_shape,
                                    _label_cube_dim, _n_output_classes);

      indices.insert(indices.end(), labels.begin(), labels.end());

      float label_val = 1.0 / labels.size();
      for (size_t j = 0; j < labels.size(); j++) {
        values.push_back(label_val);
      }

      lens.push_back(labels.size());
    }

    auto tensor = Tensor::sparse(std::move(indices), std::move(values),
                                 std::move(lens), _n_output_classes);

    label_batches.push_back({tensor});
  }

  return label_batches;
}

std::pair<ModelPtr, ComputationPtr> SeismicEmbeddingModel::buildModel(
    size_t n_patches, size_t patch_dim, size_t embedding_dim,
    size_t n_output_classes) {
  auto patches = Input::make(n_patches * patch_dim);

  size_t patch_emb_dim = 100000;
  auto patch_emb =
      PatchEmbedding::make(
          /*emb_dim=*/patch_emb_dim, /*patch_dim=*/patch_dim,
          /*n_patches=*/n_patches, /*sparsity=*/0.01, /*activation=*/"relu")
          ->apply(patches);

  auto aggregated_embs = PatchSum::make(/*n_patches=*/n_patches,
                                        /*patch_dim=*/patch_emb_dim)
                             ->apply(patch_emb);

  auto emb = FullyConnected::make(
                 /*dim=*/embedding_dim, /*input_dim=*/aggregated_embs->dim(),
                 /* sparsity=*/1.0, /*activation=*/"tanh")
                 ->apply(aggregated_embs);

  auto output =
      FullyConnected::make(
          /*dim=*/n_output_classes, /*input_dim=*/emb->dim(), /*sparsity=*/0.05,
          /*activation=*/"softmax")
          ->apply(emb);

  auto loss =
      CategoricalCrossEntropy::make(output, Input::make(n_output_classes));

  auto model = Model::make({patches}, {output}, {loss});

  return {model, emb};
}

void SeismicEmbeddingModel::save(const std::string& filename) const {
  auto output = dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(output);
}

void SeismicEmbeddingModel::save_stream(std::ostream& output) const {
  cereal::BinaryOutputArchive oarchive(output);
  _model->setSerializeOptimizer(/* should_save_optimizer= */ true);
  oarchive(*this);
}

std::shared_ptr<SeismicEmbeddingModel> SeismicEmbeddingModel::load(
    const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(input_stream);
}

std::shared_ptr<SeismicEmbeddingModel> SeismicEmbeddingModel::load_stream(
    std::istream& input) {
  cereal::BinaryInputArchive iarchive(input);
  std::shared_ptr<SeismicEmbeddingModel> deserialize_into(
      new SeismicEmbeddingModel());
  iarchive(*deserialize_into);

  return deserialize_into;
}

template <class Archive>
void SeismicEmbeddingModel::serialize(Archive& archive) {
  archive(_model, _emb, _subcube_shape, _patch_shape, _max_pool,
          _label_cube_dim, _n_output_classes);
}

}  // namespace thirdai::bolt::seismic