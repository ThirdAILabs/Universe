#include "Shuffler.h"
#include <bolt_vector/src/BoltVector.h>
#include <vector>

namespace thirdai::dataset {

void Shuffler::add(std::vector<BoltBatch>&& batch) {
  _buffer_size += batch.front().getBatchSize();
  _offsets.push_back(_buffer_size);
  _buffer.push_back(std::move(batch));
}

std::vector<BoltDatasetPtr> Shuffler::datasets(uint32_t batch_size,
                                               uint32_t max_batches) {
  std::cout << "start datasetss()" << std::endl;
  // Equivalent to vector of bolt datasets
  std::vector<std::vector<BoltBatch>> shuffled_batches =
      shuffle(std::move(_buffer), batch_size);
  std::cout << "finished shuffling" << std::endl;

  uint32_t num_returned =
      std::min<uint32_t>(max_batches, shuffled_batches.front().size());

  std::vector<BoltDatasetPtr> output(shuffled_batches.size());
  std::cout << "allocating output" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  for (uint32_t dataset_id = 0; dataset_id < output.size(); dataset_id++) {
    std::vector<BoltBatch> batches(num_returned);
    // batches.reserve(num_returned);
    std::move(shuffled_batches[dataset_id].begin(),
              shuffled_batches[dataset_id].begin() + num_returned,
              batches.begin());
    output[dataset_id] = std::make_shared<BoltDataset>(std::move(batches));
  }
  std::cout << "Moved batch list to datasets" << std::endl;
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Took " << duration << " seconds." << std::endl;

  _buffer.clear();
  _buffer_size = 0;
  _offsets = {0};
  for (uint32_t remain_id = num_returned;
       remain_id < shuffled_batches.front().size(); remain_id++) {
    std::cout << "REMAIN ID " << remain_id << std::endl;
    std::vector<BoltBatch> batch(shuffled_batches[remain_id].size());
    for (uint32_t column_id = 0; column_id < batch.size(); column_id++) {
      batch[column_id] = std::move(shuffled_batches[column_id][remain_id]);
    }
    _buffer.push_back(std::move(batch));
    _buffer_size += _buffer.front().back().getBatchSize();
    _offsets.push_back(_buffer_size);
  }
  std::cout << "Moved remains" << std::endl;
  return output;
}

std::vector<std::vector<BoltBatch>> Shuffler::shuffle(
    std::vector<std::vector<BoltBatch>>&& buffer, uint32_t batch_size) {
  std::cout << "Start shuffle" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<uint32_t> permutation(_buffer_size);
  std::cout << "Allocated permutation vector" << std::endl;
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Took " << duration << " seconds." << std::endl;
  start = end;
  std::iota(permutation.begin(), permutation.end(), 0);
  std::cout << "permutation indices filled" << std::endl;
  end = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Took " << duration << " seconds." << std::endl;
  start = end;
  if (_shuffle) {
    std::shuffle(permutation.begin(), permutation.end(), _gen);
  }
  std::cout << "shuffled permutations" << std::endl;
  end = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Took " << duration << " seconds." << std::endl;
  start = end;

  uint32_t n_columns = buffer.front().size();
  uint32_t n_shuffled_batches = (_buffer_size + batch_size - 1) / batch_size;
  uint32_t last_batch_size = _buffer_size % batch_size;

  std::vector<std::vector<BoltBatch>> shuffled_batches(
      n_columns,
      std::vector<BoltBatch>(n_shuffled_batches, BoltBatch(batch_size)));
  std::cout << "Allocated shuffled batches" << std::endl;
  end = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Took " << duration << " seconds." << std::endl;
  start = end;

  for (auto& batch_list : shuffled_batches) {
    batch_list.back() = BoltBatch(last_batch_size);
  }
  std::cout << "Allocated BoltBatches" << std::endl;
  end = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Took " << duration << " seconds." << std::endl;
  start = end;

#pragma omp parallel for default(none) \
    shared(buffer, shuffled_batches, permutation, batch_size, std::cout)
  for (uint32_t batch_id = 0; batch_id < buffer.size(); batch_id++) {
    for (uint32_t column_id = 0; column_id < buffer[batch_id].size();
         column_id++) {
      auto& unshuffled_batch = buffer[batch_id][column_id];
      for (uint32_t vec_id = 0; vec_id < unshuffled_batch.getBatchSize();
           vec_id++) {
        uint32_t sample_id = _offsets[batch_id] + vec_id;
        uint32_t shuffled_sample_id = permutation[sample_id];
        uint32_t shuffled_batch_id = shuffled_sample_id / batch_size;
        uint32_t shuffled_vec_id = shuffled_sample_id % batch_size;
        shuffled_batches[column_id][shuffled_batch_id][shuffled_vec_id] =
            std::move(buffer[batch_id][column_id][vec_id]);
      }
    }
  }
  std::cout << "Did actual shuffling" << std::endl;
  end = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Took " << duration << " seconds." << std::endl;
  start = end;

  return shuffled_batches;
}
}  // namespace thirdai::dataset