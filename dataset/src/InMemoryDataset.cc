
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
