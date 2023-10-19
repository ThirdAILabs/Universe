#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/seismic/InputShapeData.h>
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

using NumpyArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

class SeismicBase {
 public:
  SeismicBase(InputShapeData input_shape_data, ModelPtr model);

  metrics::History trainOnPatches(
      const NumpyArray& subcubes, Dataset&& labels, float learning_rate,
      size_t batch_size, const std::vector<callbacks::CallbackPtr>& callbacks,
      std::optional<uint32_t> log_interval, const DistributedCommPtr& comm);

  NumpyArray embeddingsForPatches(const NumpyArray& subcubes);

  const Shape& subcubeShape() const { return _input_shape_data.subcubeShape(); }

  const Shape& patchShape() const { return _input_shape_data.patchShape(); }

  const std::optional<Shape>& maxPool() const {
    return _input_shape_data.maxPool();
  }

  const InputShapeData& inputShapeData() const { return _input_shape_data; }

  ModelPtr getModel() const { return _model; }

  void setModel(ModelPtr model) {
    _model = std::move(model);
    auto computations = _model->computationOrderWithoutInputs();
    _emb = computations.at(computations.size() - 2);
  }

  size_t labelDim() const {
    auto label_dims = _model->labelDims();
    if (label_dims.size() != 1) {
      throw std::invalid_argument(
          "Expected model to only have 1 output layer.");
    }
    return label_dims[0];
  }

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output) const;

  static std::shared_ptr<SeismicBase> load(const std::string& filename);

  static std::shared_ptr<SeismicBase> load_stream(std::istream& input);

  virtual ~SeismicBase() = default;

 private:
  Dataset convertToBatches(const NumpyArray& array, size_t batch_size) const;

  static std::pair<ModelPtr, ComputationPtr> buildModel(size_t n_patches,
                                                        size_t patch_dim,
                                                        size_t embedding_dim,
                                                        size_t n_output_classes,
                                                        float output_sparsity);

  ModelPtr _model;
  ComputationPtr _emb;

  InputShapeData _input_shape_data;

  SeismicBase() : _input_shape_data({}, {}, {}) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

class Checkpoint final : public callbacks::Callback {
 public:
  Checkpoint(std::shared_ptr<SeismicBase> seismic_model,
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

  std::shared_ptr<SeismicBase> _seismic_model;
  std::filesystem::path _checkpoint_dir;
  size_t _interval;
};

}  // namespace thirdai::bolt::seismic