#include "Dataset.h"
#include "batch_types/DenseBatch.h"
#include "batch_types/SparseBatch.h"

namespace thirdai::utils {

template class InMemoryDataset<SparseBatch>;
template class InMemoryDataset<DenseBatch>;

template <typename BATCH_T>
InMemoryDataset<BATCH_T>::InMemoryDataset(const std::string& filename,
                                          uint32_t batch_size) {
  std::ifstream file(filename);

  uint64_t curr_id = 0;
  while (!file.eof()) {
    _batches.push_back(BATCH_T(file, batch_size, curr_id));
    curr_id += _batches.back().getBatchSize();
  }

  file.close();
}

template class StreamedDataset<SparseBatch>;
template class StreamedDataset<DenseBatch>;

template <typename BATCH_T>
std::optional<BATCH_T> StreamedDataset<BATCH_T>::nextBatch() {
  if (_file.eof()) {
    return std::nullopt;
  }

  BATCH_T next(_file, _batch_size, _curr_id);
  _curr_id += next.getBatchSize();

  return next;
}

}  // namespace thirdai::utils