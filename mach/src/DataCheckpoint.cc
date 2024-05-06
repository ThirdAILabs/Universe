#include "DataCheckpoint.h"
#include <data/src/ColumnMapIterator.h>
#include <data/src/transformations/StringCast.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <nlohmann/json.hpp>
#include <filesystem>

using json = nlohmann::json;

namespace thirdai::mach {

DataCheckpoint::DataCheckpoint(data::ColumnMapIteratorPtr data_iter,
                               std::string id_col,
                               std::vector<std::string> text_cols)
    : _data_iter(std::move(data_iter)),
      _id_col(std::move(id_col)),
      _text_cols(std::move(text_cols)) {}

void saveDataset(const data::ColumnMapIteratorPtr& data_iter,
                 const std::string& path, const std::string& id_col,
                 const std::vector<std::string>& text_cols) {
  auto output = dataset::SafeFileIO::ofstream(path);

  output << id_col;

  for (const auto& col : text_cols) {
    output << "," << col;
  }
  output << std::endl;

  while (auto chunk = data_iter->next()) {
    auto ids = chunk->getArrayColumn<uint32_t>(id_col);

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
}

std::string dataMetadataPath(const std::string& ckpt_dir) {
  return std::filesystem::path(ckpt_dir) / "metadata";
}

void DataCheckpoint::save(const std::string& ckpt_dir) {
  if (!std::filesystem::exists(ckpt_dir)) {
    std::filesystem::create_directories(ckpt_dir);
  }
  if (!_dataset_path) {
    _dataset_path = std::filesystem::path(ckpt_dir) / "dataset.csv";
    saveDataset(_data_iter, _dataset_path.value(), _id_col, _text_cols);
  }

  json metadata;

  metadata["dataset_path"] = _dataset_path.value();
  metadata["id_col"] = _id_col;
  metadata["text_cols"] = _text_cols;

  auto output = dataset::SafeFileIO::ofstream(dataMetadataPath(ckpt_dir));
  output << std::setw(4) << metadata;
}

DataCheckpoint DataCheckpoint::load(const std::string& ckpt_dir) {
  return DataCheckpoint(ckpt_dir);
}

DataCheckpoint::DataCheckpoint(const std::string& ckpt_dir) {
  auto input = dataset::SafeFileIO::ifstream(dataMetadataPath(ckpt_dir));
  json metadata;

  input >> metadata;

  _dataset_path = metadata["dataset_path"];
  _id_col = metadata["id_col"];
  _text_cols = metadata["text_cols"];

  auto iter = std::make_shared<data::CsvIterator>(_dataset_path.value(), ',');

  auto parse_labels = std::make_shared<data::StringToTokenArray>(
      _id_col, _id_col, ':', std::numeric_limits<uint32_t>::max());

  _data_iter = data::TransformedIterator::make(iter, parse_labels, nullptr);
}

}  // namespace thirdai::mach