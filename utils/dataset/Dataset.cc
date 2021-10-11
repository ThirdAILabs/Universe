#include "Dataset.h"
#include "batch_types/CsvBatch.h"
#include "batch_types/SvmBatch.h"

namespace thirdai::utils {

template class InMemoryDataset<SvmBatch>;
template class InMemoryDataset<CsvBatch>;

template <typename Batch_t>
InMemoryDataset<Batch_t>::InMemoryDataset(const std::string& filename,
                                          uint32_t batch_size) {
  std::ifstream file(filename);

  uint64_t curr_id = 0;
  while (!file.eof()) {
    _batches.push_back(Batch_t(file, batch_size, curr_id));
    curr_id += _batches.back().getBatchSize();
  }

  file.close();
}

template class StreamedDataset<SvmBatch>;
template class StreamedDataset<CsvBatch>;

template <typename Batch_t>
std::optional<Batch_t> StreamedDataset<Batch_t>::nextBatch() {
  if (_file.eof()) {
    return std::nullopt;
  }

  Batch_t next(_file, _batch_size, _curr_id);
  _curr_id += next.getBatchSize();

  return next;
}

}  // namespace thirdai::utils