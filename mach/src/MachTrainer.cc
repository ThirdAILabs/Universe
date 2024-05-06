#include "MachTrainer.h"
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/callbacks/Overfitting.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/columns/Column.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <mach/src/MachConfig.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace thirdai::mach {

class MachCheckpoint final : public bolt::callbacks::Callback {
 public:
  MachCheckpoint(MachTrainer* trainer, std::string save_path)
      : _trainer(trainer), _save_path(std::move(save_path)) {}

  void onEpochEnd() final { _trainer->makeCheckpoint(_save_path); }

 private:
  MachTrainer* _trainer;

  std::string _save_path;
};

std::string modelPath(const std::string& ckpt_dir) {
  return std::filesystem::path(ckpt_dir) / "model";
}

std::string metadataPath(const std::string& ckpt_dir) {
  return std::filesystem::path(ckpt_dir) / "metadata";
}

std::string dataPath(const std::string& ckpt_dir) {
  return std::filesystem::path(ckpt_dir) / "data";
}

MachTrainer::MachTrainer(MachRetrieverPtr model, DataCheckpoint data)
    : _model(std::move(model)),
      _initial_model_epochs(_model->model()->epochs()),
      _data_ckpt(std::move(data)) {}

MachRetrieverPtr MachTrainer::complete(
    const std::optional<std::string>& ckpt_dir) {
  auto data = _data_ckpt.data();
  if (_model->model()->trainSteps() > 0 && isColdstart() && !_loaded_ckpt) {
    // TODO(Nicholas) what should these args default to, do we need option to
    // override?
    _model->introduceIterator(data, _strong_cols, _weak_cols,
                              /*text_augmentation=*/true,
                              /*n_buckets_to_sample_opt=*/std::nullopt,
                              /*n_random_hashes=*/0, /*load_balancing=*/true,
                              /*sort_random_hashes=*/false);
    data->restart();
  }

  if (ckpt_dir) {
    if (std::filesystem::exists(*ckpt_dir)) {
      throw std::invalid_argument("Found existing checkpoint in '" + *ckpt_dir +
                                  "'.");
    }
    std::filesystem::create_directories(*ckpt_dir);
    makeCheckpoint(*ckpt_dir);
  }

  std::vector<bolt::callbacks::CallbackPtr> callbacks;
  if (_min_epochs < _max_epochs) {
    callbacks.push_back(std::make_shared<bolt::callbacks::Overfitting>(
        "train_" + _early_stop_metric, /*threshold=*/_early_stop_threshold,
        /*freeze_hash_tables=*/false, /*maximize=*/true,
        /*min_epochs=*/_initial_model_epochs + _min_epochs));

    if (std::find(_metrics.begin(), _metrics.end(), _early_stop_metric) ==
        _metrics.end()) {
      _metrics.push_back(_early_stop_metric);
    }
  }
  if (ckpt_dir) {
    callbacks.push_back(std::make_shared<MachCheckpoint>(this, *ckpt_dir));
  }

  if (isColdstart()) {
    ColdStartOptions options;
    options.batch_size = _batch_size;
    options.max_in_memory_batches = _max_in_memory_batches;

    _model->coldstart(data, _strong_cols, _weak_cols, _learning_rate,
                      epochsRemaining(_max_epochs), _metrics, callbacks,
                      options);
  } else {
    TrainOptions options;
    options.batch_size = _batch_size;
    options.max_in_memory_batches = _max_in_memory_batches;
    _model->train(data, _learning_rate, epochsRemaining(_max_epochs), _metrics,
                  callbacks, options);
  }

  return _model;
}

void MachTrainer::makeCheckpoint(const std::string& ckpt_dir) {
  const std::string model_path = modelPath(ckpt_dir);
  _model->save(model_path + "_tmp", /*with_optimizer=*/true);
  std::filesystem::rename(model_path + "_tmp", model_path);

  const std::string metadata_path = metadataPath(ckpt_dir);
  saveTrainerMetadata(metadata_path + "_tmp");
  std::filesystem::rename(metadata_path + "_tmp", metadata_path);

  _data_ckpt.save(dataPath(ckpt_dir));
}

std::shared_ptr<MachTrainer> MachTrainer::fromCheckpoint(
    const std::string& dir) {
  auto model = MachRetriever::load(modelPath(dir));

  auto data = DataCheckpoint::load(dataPath(dir));

  auto trainer =
      std::make_shared<MachTrainer>(std::move(model), std::move(data));

  trainer->loadTrainerMetadata(metadataPath(dir));

  return trainer;
}

void MachTrainer::saveTrainerMetadata(const std::string& path) const {
  json metadata;

  metadata["strong_cols"] = _strong_cols;
  metadata["weak_cols"] = _weak_cols;

  metadata["learning_rate"] = _learning_rate;
  metadata["min_epochs"] = _min_epochs;
  metadata["max_epochs"] = _max_epochs;
  metadata["initial_model_epochs"] = _initial_model_epochs;
  metadata["metrics"] = _metrics;

  if (_max_in_memory_batches) {
    metadata["max_in_memory_batches"] = *_max_in_memory_batches;
  }

  metadata["batch_size"] = _batch_size;

  metadata["early_stop_metric"] = _early_stop_metric;
  metadata["early_stop_threshold"] = _early_stop_threshold;

  auto output = dataset::SafeFileIO::ofstream(path);

  output << std::setw(4) << metadata << std::endl;
}

void MachTrainer::loadTrainerMetadata(const std::string& path) {
  auto input = dataset::SafeFileIO::ifstream(path);

  json metadata;
  input >> metadata;

  _strong_cols = metadata["strong_cols"];
  _weak_cols = metadata["weak_cols"];

  _learning_rate = metadata["learning_rate"];
  _min_epochs = metadata["min_epochs"];
  _max_epochs = metadata["max_epochs"];
  _initial_model_epochs = metadata["initial_model_epochs"];
  _metrics = metadata["metrics"];
  if (metadata.contains("max_in_memory_batches")) {
    _max_in_memory_batches = metadata["max_in_memory_batches"];
  }
  _batch_size = metadata["batch_size"];

  _early_stop_metric = metadata["early_stop_metric"];
  _early_stop_threshold = metadata["early_stop_threshold"];

  _loaded_ckpt = true;
}

MachTrainer& MachTrainer::strongWeakCols(
    const std::vector<std::string>& strong_cols,
    const std::vector<std::string>& weak_cols) {
  _strong_cols = strong_cols;
  _weak_cols = weak_cols;
  return *this;
}

MachTrainer& MachTrainer::learningRate(float learning_rate) {
  _learning_rate = learning_rate;
  return *this;
}

MachTrainer& MachTrainer::minMaxEpochs(uint32_t min_epochs,
                                       uint32_t max_epochs) {
  _min_epochs = min_epochs;
  _max_epochs = max_epochs;
  return *this;
}

MachTrainer& MachTrainer::metrics(const std::vector<std::string>& metrics) {
  _metrics = metrics;
  return *this;
}

MachTrainer& MachTrainer::maxInMemoryBatches(
    std::optional<uint32_t> max_in_memory_batches) {
  _max_in_memory_batches = max_in_memory_batches;
  return *this;
}

MachTrainer& MachTrainer::batchSize(uint32_t batch_size) {
  _batch_size = batch_size;
  return *this;
}

MachTrainer& MachTrainer::earlyStop(const std::string& metric,
                                    float threshold) {
  _early_stop_metric = metric;
  _early_stop_threshold = threshold;
  return *this;
}

}  // namespace thirdai::mach