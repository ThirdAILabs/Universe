#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/seismic/SeismicLabels.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/DistributedComm.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <filesystem>
#include <istream>

namespace py = pybind11;

namespace thirdai::bolt::seismic {

using Shape = std::tuple<size_t, size_t, size_t>;

using NumpyArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

class SeismicEmbeddingModel {
 public:
  SeismicEmbeddingModel(size_t subcube_shape, size_t patch_shape,
                        size_t embedding_dim, std::optional<size_t> max_pool);

  metrics::History trainOnPatches(
      const NumpyArray& subcubes,
      const std::vector<SubcubeMetadata>& subcube_metadata, float learning_rate,
      size_t batch_size, const std::vector<callbacks::CallbackPtr>& callbacks,
      std::optional<uint32_t> log_interval, const DistributedCommPtr& comm);

  NumpyArray embeddingsForPatches(const NumpyArray& subcubes);

  auto subcubeShape() const { return _subcube_shape; }

  auto patchShape() const { return _patch_shape; }

  auto maxPool() const { return _max_pool; }

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output) const;

  static std::shared_ptr<SeismicEmbeddingModel> load(
      const std::string& filename);

  static std::shared_ptr<SeismicEmbeddingModel> load_stream(
      std::istream& input);

  ModelPtr getModel() const { return _model; }

  void setModel(ModelPtr model) {
    _model = std::move(model);
    auto computations = _model->computationOrderWithoutInputs();
    _emb = computations.at(computations.size() - 2);
  }

 private:
  Dataset convertToBatches(const NumpyArray& array, size_t batch_size) const;

  Dataset makeLabelBatches(const std::vector<SubcubeMetadata>& subcube_metadata,
                           size_t batch_size) const;

  size_t nPatches() const {
    auto [dim_x, dim_y, dim_z] = _subcube_shape;
    auto [patch_x, patch_y, patch_z] = _patch_shape;
    return (dim_x / patch_x) * (dim_y / patch_y) * (dim_z / patch_z);
  }

  size_t flattenedPatchDim() const {
    auto [patch_x, patch_y, patch_z] = _patch_shape;
    if (_max_pool) {
      auto [pool_x, pool_y, pool_z] = *_max_pool;
      return (patch_x / pool_x) * (patch_y / pool_y) * (patch_z / pool_z);
    }
    return patch_x * patch_y * patch_z;
  }

  static std::pair<ModelPtr, ComputationPtr> buildModel(
      size_t n_patches, size_t patch_dim, size_t embedding_dim,
      size_t n_output_classes);

  ModelPtr _model;
  ComputationPtr _emb;

  // These shapes are stored as tuples because we want to support a case  where
  // the subcubes are 2D, with a shape like (1, 10, 10), but we sill want these
  // 2D slices that are nearby in the x-axis to share labels. Thus the labels
  // are always associated with 3D cubes in space, which ensures that
  // overlapping subcubes in any of the three axes will share labels.
  Shape _subcube_shape;
  Shape _patch_shape;
  std::optional<Shape> _max_pool;

  // TODO(Nicholas): support for list of label cube dims for different
  // granularities.
  size_t _label_cube_dim = 32;

  size_t _n_output_classes = 50000;

  SeismicEmbeddingModel() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

class Checkpoint final : public callbacks::Callback {
 public:
  Checkpoint(std::shared_ptr<SeismicEmbeddingModel> seismic_model,
             const std::string& checkpoint_dir, size_t interval)
      : _seismic_model(std::move(seismic_model)),
        _checkpoint_dir(checkpoint_dir),
        _interval(interval) {}

  void onBatchEnd() final {
    if ((model->trainSteps() % _interval) == (_interval - 1)) {
      _seismic_model->save(checkpointPath());
    }
  }

  void onEpochEnd() final {
    _seismic_model->save(checkpointPath() + "_epoch_end");
  }

 private:
  std::string checkpointPath() const {
    std::filesystem::path ckpt = "step_" + std::to_string(model->trainSteps());
    return (_checkpoint_dir / ckpt).string();
  }

  std::shared_ptr<SeismicEmbeddingModel> _seismic_model;
  std::filesystem::path _checkpoint_dir;
  size_t _interval;
};

}  // namespace thirdai::bolt::seismic