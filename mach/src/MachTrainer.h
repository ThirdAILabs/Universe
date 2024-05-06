#pragma once

#include <data/src/ColumnMapIterator.h>
#include <mach/src/DataCheckpoint.h>
#include <mach/src/MachConfig.h>
#include <mach/src/MachRetriever.h>
#include <optional>

namespace thirdai::mach {

class MachTrainer {
 public:
  explicit MachTrainer(MachRetrieverPtr model, DataCheckpoint data);

  MachRetrieverPtr complete(const std::optional<std::string>& ckpt_dir);

  void makeCheckpoint(const std::string& ckpt_dir);

  static std::shared_ptr<MachTrainer> fromCheckpoint(const std::string& dir);

  MachTrainer& strongWeakCols(const std::vector<std::string>& strong_cols,
                              const std::vector<std::string>& weak_cols);

  MachTrainer& learningRate(float learning_rate);

  MachTrainer& minMaxEpochs(uint32_t min_epochs, uint32_t max_epochs);

  MachTrainer& metrics(const std::vector<std::string>& metrics);

  MachTrainer& maxInMemoryBatches(
      std::optional<uint32_t> max_in_memory_batches);

  MachTrainer& batchSize(uint32_t batch_size);

  MachTrainer& earlyStop(const std::string& metric, float threshold);

 private:
  bool isColdstart() const {
    return !_strong_cols.empty() || !_weak_cols.empty();
  }

  void saveTrainerMetadata(const std::string& path) const;

  void loadTrainerMetadata(const std::string& path);

  uint32_t epochsRemaining(uint32_t epochs) const {
    if (_initial_model_epochs + epochs < _model->model()->epochs()) {
      return 0;
    }
    return _initial_model_epochs + epochs - _model->model()->epochs();
  }

  MachRetrieverPtr _model;
  uint32_t _initial_model_epochs;

  DataCheckpoint _data_ckpt;

  std::vector<std::string> _strong_cols;
  std::vector<std::string> _weak_cols;

  float _learning_rate = 1e-3;
  uint32_t _min_epochs = 5;
  uint32_t _max_epochs = 10;
  std::vector<std::string> _metrics;
  std::optional<uint32_t> _max_in_memory_batches;
  uint32_t _batch_size = 2000;

  std::string _early_stop_metric = "hash_precision@5";
  float _early_stop_threshold = 0.95;

  bool _loaded_ckpt = false;
};

}  // namespace thirdai::mach