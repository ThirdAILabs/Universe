#include "MachTrainer.h"
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/callbacks/Overfitting.h>
#include <_types/_uint32_t.h>
#include <archive/src/Archive.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/StringCast.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <mach/src/MachConfig.h>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::mach {

class MachCheckpoint final : public bolt::callbacks::Callback {
 public:
  MachCheckpoint(MachTrainer* trainer, std::string save_path)
      : _trainer(trainer), _save_path(std::move(save_path)) {}

  void onEpochEnd() final { _trainer->intermediateCheckpoint(_save_path); }

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

MachRetrieverPtr MachTrainer::complete(
    const std::optional<std::string>& ckpt_dir) {
  if (_model->model()->trainSteps() > 0 && isColdstart() && !_loaded_ckpt) {
    // TODO(Nicholas) what should these args default to, do we need option to
    // override?
    _model->introduceIterator(_data, _strong_cols, _weak_cols,
                              /*text_augmentation=*/true,
                              /*n_buckets_to_sample_opt=*/std::nullopt,
                              /*n_random_hashes=*/0, /*load_balancing=*/true,
                              /*sort_random_hashes=*/false);
    _data->restart();
  }

  if (ckpt_dir) {
    initialCheckpoint(*ckpt_dir);
  }

  std::vector<bolt::callbacks::CallbackPtr> callbacks;
  if (_min_epochs < _max_epochs) {
    callbacks.push_back(std::make_shared<bolt::callbacks::Overfitting>(
        "train_" + _early_stop_metric, /*threshold=*/_early_stop_threshold,
        /*freeze_hash_tables=*/false, /*maximize=*/true,
        /*min_epochs=*/_min_epochs));

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
    options.variable_length = _vlc;

    _model->coldstart(_data, _strong_cols, _weak_cols, _learning_rate,
                      correctEpochs(_max_epochs), _metrics, callbacks, options);
  } else {
    TrainOptions options;
    options.batch_size = _batch_size;
    options.max_in_memory_batches = _max_in_memory_batches;
    _model->train(_data, _learning_rate, correctEpochs(_max_epochs), _metrics,
                  callbacks, options);
  }

  return _model;
}

void MachTrainer::initialCheckpoint(const std::string& ckpt_dir) const {
  if (std::filesystem::exists(ckpt_dir)) {
    throw std::invalid_argument("Found existing checkpoint in '" + ckpt_dir +
                                "'.");
  }

  std::filesystem::create_directories(ckpt_dir);

  _model->save(modelPath(ckpt_dir), /*with_optimizer=*/true);

  saveTrainerMetadata(metadataPath(ckpt_dir));

  saveDataset(dataPath(ckpt_dir));
}

void MachTrainer::intermediateCheckpoint(const std::string& ckpt_dir) {
  _model->save(modelPath(ckpt_dir), /*with_optimizer=*/true);

  saveTrainerMetadata(metadataPath(ckpt_dir));
}

std::shared_ptr<MachTrainer> MachTrainer::fromCheckpoint(
    const std::string& dir) {
  auto model = MachRetriever::load(modelPath(dir));

  auto data = loadDataset(dataPath(dir), model->idCol());

  auto trainer =
      std::make_shared<MachTrainer>(std::move(model), std::move(data));

  trainer->loadTrainerMetadata(metadataPath(dir));

  return trainer;
}

void MachTrainer::saveTrainerMetadata(const std::string& path) const {
  auto map = ar::Map::make();

  map->set("strong_cols", ar::vecStr(_strong_cols));
  map->set("weak_cols", ar::vecStr(_weak_cols));
  if (_vlc) {
    map->set("vlc", _vlc->toArchive());
  }

  map->set("learning_rate", ar::f32(_learning_rate));
  map->set("min_epochs", ar::u64(_min_epochs));
  map->set("max_epochs", ar::u64(_max_epochs));
  map->set("metrics", ar::vecStr(_metrics));
  if (_max_in_memory_batches) {
    map->set("max_in_memory_batches", ar::u64(*_max_in_memory_batches));
  }
  map->set("batch_size", ar::u64(_batch_size));

  map->set("early_stop_metric", ar::str(_early_stop_metric));
  map->set("early_stop_threshold", ar::f32(_early_stop_threshold));

  auto output = dataset::SafeFileIO::ofstream(path);
  ar::serialize(map, output);
}

void MachTrainer::loadTrainerMetadata(const std::string& path) {
  auto input = dataset::SafeFileIO::ifstream(path);
  auto archive = ar::deserialize(input);

  _strong_cols = archive->getAs<ar::VecStr>("strong_cols");
  _weak_cols = archive->getAs<ar::VecStr>("weak_cols");
  if (archive->contains("vlc")) {
    _vlc = data::VariableLengthConfig(*archive->get("vlc"));
  }

  _learning_rate = archive->f32("learning_rate");
  _min_epochs = archive->u64("min_epochs");
  _max_epochs = archive->u64("max_epochs");
  _metrics = archive->getAs<ar::VecStr>("metrics");
  _max_in_memory_batches = archive->getOpt<ar::U64>("max_in_memory_batches");
  _batch_size = archive->u64("batch_size");

  _early_stop_metric = archive->str("early_stop_metric");
  _early_stop_threshold = archive->f32("early_stop_threshold");

  _loaded_ckpt = true;
}

void MachTrainer::saveDataset(const std::string& path) const {
  auto output = dataset::SafeFileIO::ofstream(path);

  output << _model->idCol();

  std::vector<std::string> text_cols;
  if (isColdstart()) {
    text_cols.insert(text_cols.end(), _strong_cols.begin(), _strong_cols.end());
    text_cols.insert(text_cols.end(), _weak_cols.begin(), _weak_cols.end());
  } else {
    text_cols.push_back(_model->textCol());
  }

  for (const auto& col : text_cols) {
    output << "," << col;
  }
  output << std::endl;

  while (auto chunk = _data->next()) {
    auto ids = chunk->getArrayColumn<uint32_t>(_model->idCol());

    std::vector<data::ValueColumnBasePtr<std::string>> texts;
    texts.reserve(text_cols.size());
    for (const auto& col : text_cols) {
      texts.push_back(chunk->getValueColumn<std::string>(col));
    }

    for (size_t i = 0; i < ids->numRows(); i++) {
      auto row_ids = ids->row(i);
      output << row_ids[0];
      for (size_t j = 1; j < row_ids.size(); j++) {
        output << ":" << row_ids[j];
      }

      for (const auto& text : texts) {
        output << ",\"" << text->value(i) << '"';
      }
      output << std::endl;
    }
  }

  _data->restart();
}

data::ColumnMapIteratorPtr MachTrainer::loadDataset(const std::string& path,
                                                    const std::string& id_col) {
  auto iter = std::make_shared<data::CsvIterator>(path, ',');

  auto parse_labels = std::make_shared<data::StringToTokenArray>(
      id_col, id_col, ':', std::numeric_limits<uint32_t>::max());

  return data::TransformedIterator::make(iter, parse_labels, nullptr);
}

MachTrainer& MachTrainer::strongWeakCols(
    const std::vector<std::string>& strong_cols,
    const std::vector<std::string>& weak_cols) {
  _strong_cols = strong_cols;
  _weak_cols = weak_cols;
  return *this;
}

MachTrainer& MachTrainer::vlc(
    const std::optional<data::VariableLengthConfig>& vlc) {
  _vlc = vlc;
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