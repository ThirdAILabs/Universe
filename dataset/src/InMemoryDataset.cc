
#include "InMemoryDataset.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

namespace thirdai::dataset {

InMemoryDataset::InMemoryDataset(std::vector<BoltBatch>&& batches)
    : _batches(std::move(batches)) {
  if (_batches.empty()) {
    throw std::invalid_argument(
        "Must pass in at least one batch to the dataset constructor but "
        "found 0.");
  }
  _batch_size = _batches.front().getBatchSize();
  if (_batch_size == 0) {
    throw std::invalid_argument(
        "The first batch was found to have an invalid length of 0.");
  }

  for (uint64_t i = 1; i < _batches.size() - 1; i++) {
    uint64_t current_batch_size = _batches.at(i).getBatchSize();
    if (current_batch_size != _batch_size) {
      std::cout << current_batch_size << " " << _batch_size << std::endl;
      throw std::invalid_argument(
          "All batches but the last batch must have the same size.");
    }
  }

  uint64_t last_batch_size = _batches.back().getBatchSize();
  if (last_batch_size > _batch_size) {
    throw std::invalid_argument(
        "The last batch in the dataset is larger than the others, when it "
        "should be equal to or smaller than them in length.");
  }
  if (last_batch_size == 0) {
    throw std::invalid_argument(
        "The last batch was found to have an invalid length of 0.");
  }

  _len = _batch_size * (_batches.size() - 1) + last_batch_size;
}

template <class Archive>
void InMemoryDataset::serialize(Archive& archive) {
  archive(_batches, _len, _batch_size);
}

std::shared_ptr<InMemoryDataset> InMemoryDataset::load(
    const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::BinaryInputArchive iarchive(filestream);
  auto deserialize_into = std::make_shared<InMemoryDataset>();
  iarchive(*deserialize_into);
  return deserialize_into;
}

void InMemoryDataset::save(const std::string& filename) {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(filestream);
  oarchive(*this);
}

}  // namespace thirdai::dataset
