#include "Dataset.h"
#include "batch_types/DenseBatch.h"
#include "batch_types/SparseBatch.h"

namespace thirdai::utils {

template class InMemoryDataset<SparseBatch>;
template class InMemoryDataset<DenseBatch>;

template <typename Batch_t>
InMemoryDataset<Batch_t>::InMemoryDataset(const std::string& filename,
                                          uint32_t batch_size,
                                          BatchOptions options) {
  std::ifstream file(filename);

  uint64_t curr_id = 0;
  while (!file.eof()) {
    _batches.push_back(Batch_t(file, batch_size, curr_id, options));
    curr_id += _batches.back().getBatchSize();
  }

  file.close();
  _len = curr_id;
}

template class StreamedDataset<SparseBatch>;
template class StreamedDataset<DenseBatch>;

template <typename Batch_t>
std::optional<Batch_t> StreamedDataset<Batch_t>::nextBatch() {
  if (_file.eof()) {
    return std::nullopt;
  }

  Batch_t next(_file, _batch_size, _curr_id, _options);
  _curr_id += next.getBatchSize();

  return next;
}

}  // namespace thirdai::utils