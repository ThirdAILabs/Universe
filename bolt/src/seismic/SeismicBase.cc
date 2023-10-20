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
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt::seismic {

SeismicBase::SeismicBase(InputShapeData input_shape_data, ModelPtr model)
    : _input_shape_data(std::move(input_shape_data)) {
  setModel(std::move(model));

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

std::pair<ModelPtr, ComputationPtr> SeismicBase::buildModel(
    size_t n_patches, size_t patch_dim, size_t embedding_dim,
    size_t n_output_classes, float output_sparsity) {
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

  auto output = FullyConnected::make(
                    /*dim=*/n_output_classes, /*input_dim=*/emb->dim(),
                    /*sparsity=*/output_sparsity,
                    /*activation=*/"softmax")
                    ->apply(emb);

  auto loss =
      CategoricalCrossEntropy::make(output, Input::make(n_output_classes));

  auto model = Model::make({patches}, {output}, {loss});

  return {model, emb};
}

template void SeismicBase::serialize(cereal::BinaryInputArchive&);
template void SeismicBase::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void SeismicBase::serialize(Archive& archive) {
  archive(_model, _emb, _input_shape_data);
}

}  // namespace thirdai::bolt::seismic