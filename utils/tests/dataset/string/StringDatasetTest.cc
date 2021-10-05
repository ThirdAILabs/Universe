#include "../../../dataset/string/StringDataset.h"
#include <gtest/gtest.h>

namespace thirdai::utils {

static uint64_t get_expected_num_batches(uint64_t target_batch_size,
                                         uint64_t target_batch_number,
                                         uint64_t number_of_times_loaded,
                                         uint64_t vec_num) {
  uint64_t total_batch_count =
      (vec_num + target_batch_size - 1) / target_batch_size;
  if (target_batch_number == 0) {
    return total_batch_count;
  }
  return std::min(
      target_batch_number,
      total_batch_count - ((number_of_times_loaded - 1) * target_batch_number));
}

static uint64_t get_expected_batch_size(uint64_t target_batch_size,
                                        uint64_t target_batch_number,
                                        uint64_t number_of_times_loaded,
                                        uint64_t vec_num,
                                        uint64_t batch_i_in_load) {
  // Only the last batch can have size < target_batch_size. So assume all
  // previous batches have size = target_batch_size.
  return std::min(target_batch_size, vec_num -
                                         (number_of_times_loaded - 1) *
                                             target_batch_number *
                                             target_batch_size -
                                         batch_i_in_load * target_batch_size);
}

static void evaluate_load(StringDataset& data, uint64_t target_batch_size,
                          uint64_t target_batch_number,
                          uint64_t number_of_times_loaded, uint64_t vec_num,
                          uint64_t& expected_starting_id) {
  if (target_batch_number > 0) {
    if (data.numBatches() > target_batch_number) {
      std::cout << "Num batches is greater than target batch number. Something "
                   "is terribly wrong."
                << std::endl;
    }
    ASSERT_LE(data.numBatches(), target_batch_number);
  }
  uint64_t expected_num_batches = get_expected_num_batches(
      target_batch_size, target_batch_number, number_of_times_loaded, vec_num);
  if (data.numBatches() != expected_num_batches) {
    std::cout << "Num batches expected: " << expected_num_batches
              << " got: " << data.numBatches() << std::endl
              << " Config: bn = " << target_batch_number
              << " bs = " << target_batch_size
              << " successful loads = " << number_of_times_loaded << std::endl;
  }
  ASSERT_EQ(data.numBatches(), expected_num_batches);
  for (size_t batch_i = 0; batch_i < data.numBatches(); batch_i++) {
    ASSERT_LE(data[batch_i]._batch_size, target_batch_size);
    uint64_t expected_batch_size =
        get_expected_batch_size(target_batch_size, target_batch_number,
                                number_of_times_loaded, vec_num, batch_i);
    if (data[batch_i]._batch_size != expected_batch_size) {
      std::cout << "Batch size expected: " << expected_batch_size
                << " got: " << data[batch_i]._batch_size << std::endl
                << " Config: bn = " << target_batch_number
                << " bs = " << target_batch_size
                << " successful loads = " << number_of_times_loaded
                << " batch_i = " << batch_i << std::endl;
    }
    ASSERT_EQ(data[batch_i]._batch_size, expected_batch_size);
    if (data[batch_i]._starting_id != expected_starting_id) {
      std::cout << "Batch starting id expected: " << expected_starting_id
                << " got: " << data[batch_i]._starting_id << std::endl
                << " Config: bn = " << target_batch_number
                << " bs = " << target_batch_size
                << " successful loads = " << number_of_times_loaded
                << " batch_i = " << batch_i << std::endl;
    }
    ASSERT_EQ(data[batch_i]._starting_id, expected_starting_id);
    expected_starting_id += expected_batch_size;
  }
}

std::string filename = "FreelandSep10_2020.txt";

TEST(StringDatasetTest, BatchesSentence) {
  // Make sure that sentence loader is working
  SentenceLoader loader;
  loader.addFileToQueue(filename);
  std::string buf;
  uint64_t expected_vec_num = 0;
  while (loader.loadNextString(buf)) {
    buf.clear();
    expected_vec_num++;
  }

  ASSERT_GT(expected_vec_num, 0);
  std::cout << "Expect " << expected_vec_num << " vectors in this dataset."
            << std::endl;

  uint64_t batch_sizes[] = {150, 100, 50, 30};
  uint64_t batch_nums[] = {150, 100, 50, 0};

  for (auto bs : batch_sizes) {
    for (auto bn : batch_nums) {
      uint64_t expected_starting_id = 0;
      StringDataset Data(FRAGMENT_TYPE::SENTENCE, bs, bn);
      Data.addFileToQueue(filename);
      size_t successful_loads = 0;

      Data.loadNextBatchSet();
      successful_loads++;
      evaluate_load(Data, bs, bn, successful_loads, expected_vec_num,
                    expected_starting_id);

      while (Data.numBatches() > 0) {
        Data.loadNextBatchSet();
        if (Data.numBatches() > 0) {
          successful_loads++;
          evaluate_load(Data, bs, bn, successful_loads, expected_vec_num,
                        expected_starting_id);
        }
      }

      // Verify that another load would lead to numbatches = 0
      Data.loadNextBatchSet();
      ASSERT_EQ(Data.numBatches(), 0);

      uint64_t expected_num_loads;

      if (bn > 0) {
        expected_num_loads = (expected_vec_num + (bs * bn) - 1) / (bs * bn);
      } else {
        expected_num_loads = 1;
      }
      if (successful_loads != expected_num_loads) {
        std::cout << "Num loads expected: " << expected_num_loads
                  << " got: " << successful_loads << std::endl
                  << " Config: bn = " << bn << " bs = " << bs << std::endl;
      }
      ASSERT_EQ(successful_loads, expected_num_loads);
    }
  }
}

}  // namespace thirdai::utils