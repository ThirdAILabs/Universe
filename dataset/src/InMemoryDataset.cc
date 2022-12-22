
#include "InMemoryDataset.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

namespace thirdai::dataset {

template <class Archive>
void DatasetBase::serialize(Archive& archive) {
  (void)archive;
}

template <class BATCH_T>
InMemoryDataset<BATCH_T>::InMemoryDataset(std::vector<BATCH_T>&& batches)
    : _batches(std::move(batches)) {
  if (_batches.empty()) {
    throw std::invalid_argument(
        "Must pass in at least one batch to the dataset constructor but "
        "found 0.");
  }
  _batch_size = _batches.front().size();
  if (_batch_size == 0) {
    throw std::invalid_argument(
        "The first batch was found to have an invalid length of 0.");
  }

  for (uint64_t i = 1; i < _batches.size() - 1; i++) {
    uint64_t current_batch_size = _batches.at(i).size();
    if (current_batch_size != _batch_size) {
      throw std::invalid_argument(
          "All batches but the last batch must have the same size.");
    }
  }

  uint64_t last_batch_size = _batches.back().size();
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

template <class BATCH_T>
template <class Archive>
void InMemoryDataset<BATCH_T>::serialize(Archive& archive) {
  archive(cereal::base_class<DatasetBase>(this), _batches, _len, _batch_size);
}

template <class BATCH_T>
std::shared_ptr<InMemoryDataset<BATCH_T>> InMemoryDataset<BATCH_T>::load(
    const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::BinaryInputArchive iarchive(filestream);
  auto deserialize_into = std::make_shared<InMemoryDataset<BATCH_T>>();
  iarchive(*deserialize_into);
  return deserialize_into;
}

template <class BATCH_T>
void InMemoryDataset<BATCH_T>::save(const std::string& filename) {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(filestream);
  oarchive(*this);
}

template class InMemoryDataset<thirdai::BoltBatch>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::InMemoryDataset<thirdai::BoltBatch>)
