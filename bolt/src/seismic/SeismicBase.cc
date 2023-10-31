#include "SeismicBase.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/tuple.hpp>
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
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
#include <versioning/src/Versions.h>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt::seismic {

SeismicBase::SeismicBase(InputShapeData input_shape_data, ModelPtr model,
                         bool embedding_last)
    : _input_shape_data(std::move(input_shape_data)) {
  setModel(std::move(model), /* embedding_last= */ embedding_last);

#if THIRDAI_EXPOSE_ALL
  _model->summary();
#endif
}

metrics::History SeismicBase::trainOnPatches(
    const NumpyArray& subcubes, Dataset&& labels, float learning_rate,
    size_t batch_size, const std::vector<callbacks::CallbackPtr>& callbacks,
    std::optional<uint32_t> log_interval, const DistributedCommPtr& comm) {
  auto data = convertToBatches(subcubes, batch_size);

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

NumpyArray SeismicBase::embeddingsForPatches(const NumpyArray& subcubes) {
  auto batch = convertToBatches(subcubes, subcubes.shape(0)).at(0);

  _model->forward(batch);

  return python::tensorToNumpy(_emb->tensor(),
                               /* single_row_to_vector= */ false);
}

Dataset SeismicBase::convertToBatches(const NumpyArray& array,
                                      size_t batch_size) const {
  size_t n_patches = _input_shape_data.nPatches();
  size_t flattened_patch_dim = _input_shape_data.flattenedPatchDim();
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

void SeismicBase::setModel(ModelPtr model, bool embedding_last) {
  _model = std::move(model);
  auto computations = _model->computationOrderWithoutInputs();
  auto new_emb =
      computations.at(computations.size() - (embedding_last ? 1 : 2));
  if (_emb && _emb->dim() != new_emb->dim()) {
    throw std::runtime_error("Cannot set a model with embedding dimension " +
                             std::to_string(new_emb->dim()) +
                             " in place of a model with embedding dimension " +
                             std::to_string(_emb->dim()));
  }
  _emb = new_emb;
}

template void SeismicBase::serialize(cereal::BinaryInputArchive&, uint32_t);
template void SeismicBase::serialize(cereal::BinaryOutputArchive&, uint32_t);

template <class Archive>
void SeismicBase::serialize(Archive& archive, uint32_t version) {
  // Adding a version in case we need to add custom logic to ensure
  // compatability in the future.
  (void)version;
  archive(_model, _emb, _input_shape_data);
}

}  // namespace thirdai::bolt::seismic

CEREAL_CLASS_VERSION(thirdai::bolt::seismic::SeismicBase,
                     thirdai::versions::SEISMIC_MODEL_VERSION)