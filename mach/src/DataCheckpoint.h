#pragma once

#include <data/src/ColumnMapIterator.h>

namespace thirdai::mach {

class DataCheckpoint {
 public:
  DataCheckpoint(data::ColumnMapIteratorPtr data_iter, std::string id_col,
                 std::vector<std::string> text_cols);

  const auto& data() const {
    _data_iter->restart();
    return _data_iter;
  }

  void save(const std::string& ckpt_dir);

  static DataCheckpoint load(const std::string& ckpt_dir);

 private:
  explicit DataCheckpoint(const std::string& ckpt_dir);

  data::ColumnMapIteratorPtr _data_iter;

  std::optional<std::string> _dataset_path;

  std::string _id_col;
  std::vector<std::string> _text_cols;
};

}  // namespace thirdai::mach