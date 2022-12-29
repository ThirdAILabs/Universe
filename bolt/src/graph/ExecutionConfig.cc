#include "ExecutionConfig.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>

namespace thirdai::bolt {

void TrainConfig::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void TrainConfig::save_stream(std::ostream& output_stream) const {
  if (_callbacks.numCallbacks() != 0) {
    throw std::runtime_error(
        "Cannot serialize a training config that has callbacks.");
  }
  if (_validation_context.has_value()) {
    throw std::runtime_error(
        "Cannot serialize a training config that has a validation context.");
  }
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

TrainConfigPtr TrainConfig::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

TrainConfigPtr TrainConfig::load_stream(std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<TrainConfig> deserialize_into(new TrainConfig());
  iarchive(*deserialize_into);
  return deserialize_into;
}

template <class Archive>
void TrainConfig::serialize(Archive& archive) {
  archive(_epochs, _learning_rate, _metric_names, _verbose,
          _rebuild_hash_tables, _reconstruct_hash_functions);
}

}  // namespace thirdai::bolt
