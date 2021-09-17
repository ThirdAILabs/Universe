#include "../../../dataset/Dataset.h"
#include "../../../dataset/svm/SVMDataset.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using thirdai::utils::Batch;
using thirdai::utils::SVMDataset;

// Test 1: check _num_batches, _batch_size, number of times until _num_batches =
// 0

// Test 2: compare written and read files

static std::string format_vector_as_svm_line(const Batch& batch,
                                             uint64_t vec_i) {
  std::stringstream line;
  line << batch._labels[vec_i][0];
  for (size_t label_i = 1; label_i < batch._label_lens[vec_i]; label_i++) {
    line << "," << batch._labels[vec_i][label_i];
  }
  if (batch._lens[vec_i] > 0) {
    line << " ";
    line << batch._indices[vec_i][0] << ":" << batch._values[vec_i][0];
  }
  for (size_t nonzero_i = 1; nonzero_i < batch._label_lens[vec_i];
       nonzero_i++) {
    line << " " << batch._labels[vec_i][nonzero_i] << ":"
         << batch._values[vec_i][nonzero_i];
  }
  line << "\n";
  return line.str();
}

static void append_batch_to_file(std::ostream& file, const Batch& batch) {
  for (size_t vec_i = 0; vec_i < batch._batch_size; vec_i++) {
    std::stringstream line;
    line << batch._labels[vec_i][0];
    for (size_t label_i = 1; label_i < batch._label_lens[vec_i]; label_i++) {
      line << "," << batch._labels[vec_i][label_i];
    }
    if (batch._lens[vec_i] > 0) {
      line << " ";
      line << batch._indices[vec_i][0] << ":" << batch._values[vec_i][0];
    }
    for (size_t nonzero_i = 1; nonzero_i < batch._label_lens[vec_i];
         nonzero_i++) {
      line << " " << batch._labels[vec_i][nonzero_i] << ":"
           << batch._values[vec_i][nonzero_i];
    }
    line << "\n";
    file << line.str();
  }
}

static uint64_t get_expected_num_batches(uint64_t target_batch_size,
                                         uint64_t target_batch_number,
                                         uint64_t number_of_times_loaded,
                                         uint64_t vec_num) {
  uint64_t total_batch_count =
      (vec_num + target_batch_size - 1) / target_batch_size;
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
  return std::min(target_batch_number, vec_num -
                                           (number_of_times_loaded - 1) *
                                               target_batch_number *
                                               target_batch_size -
                                           batch_i_in_load * target_batch_size);
}

static void evaluate_load(SVMDataset& Data, uint64_t target_batch_size,
                          uint64_t target_batch_number,
                          uint64_t number_of_times_loaded, uint64_t vec_num) {
  ASSERT_LE(Data.numBatches(), target_batch_number);
  uint64_t expected_num_batches = get_expected_num_batches(
      target_batch_size, target_batch_number, number_of_times_loaded, vec_num);
  ASSERT_EQ(Data.numBatches(), expected_num_batches);
  for (size_t batch_i = 0; batch_i < Data.numBatches(); batch_i++) {
    ASSERT_LE(Data[batch_i]._batch_size, target_batch_size);
    uint64_t expected_batch_size =
        get_expected_batch_size(target_batch_size, target_batch_number,
                                number_of_times_loaded, vec_num, batch_i);
    ASSERT_EQ(Data[batch_i]._batch_size, expected_batch_size);
  }
}

/**
 * Loads SVM dataset with different target batch size and batch number
 * combinations and checks:
 *  - The number of times loadNextBatchSet() is called.
 *  - The number of batches loaded each time loadNextBatchSet() is called.
 *  - The size of each batch.
 */
TEST(SVMDatasetTest, BatchSizeAndNumber) {
  // download from file from
  // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/bibtex.bz2,
  // then substitute with local path
  std::string filename = "/Users/benitogeordie/Downloads/bibtex";

  // count number of vectors (non-whitespace lines) in the file.
  std::ifstream _file(filename);
  std::string line;
  uint64_t expected_vec_num;
  while (std::getline(_file, line)) {
    if (line.find_first_not_of(" \t\n\v\f\r") != std::string::npos) {
      expected_vec_num++;
    }
  }
  _file.close();

  uint64_t batch_sizes[] = {50, 100, 150};
  uint64_t batch_nums[] = {50, 100, 150};

  for (auto bs : batch_sizes) {
    for (auto bn : batch_nums) {
      SVMDataset Data(filename, bs, bn);
      size_t loaded_i = 0;

      Data.loadNextBatchSet();
      loaded_i++;
      evaluate_load(Data, bs, bn, loaded_i, expected_vec_num);

      while (Data.numBatches() == bn) {
        Data.loadNextBatchSet();
        loaded_i++;
        evaluate_load(Data, bs, bn, loaded_i, expected_vec_num);
      }

      // Verify that another load would lead to numbatches = 0
      Data.loadNextBatchSet();
      ASSERT_EQ(Data.numBatches(), 0);

      uint64_t expected_num_loads =
          expected_vec_num + (bs * bn) - 1 / (bs * bn);
      ASSERT_EQ(loaded_i, expected_num_loads);
    }
  }
}

/**
 * Loads SVM dataset with different target batch size and batch number
 * combinations, converts each configuration back to an SVM file, and checks
 * that the rewritten file is the same as the initial file.
 */
TEST(SVMDatasetTest, CompareRewrittenFile) {
  // download from file from
  // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/bibtex.bz2,
  // then substitute with local path
  std::string filename = "/Users/benitogeordie/Downloads/bibtex";
  std::ifstream file(filename);
  std::string line_from_file;

  uint64_t batch_sizes[] = {50, 100, 150};
  uint64_t batch_nums[] = {50, 100, 150};
  for (auto bs : batch_sizes) {
    for (auto bn : batch_nums) {
      SVMDataset Data(filename, bs, bn);

      Data.loadNextBatchSet();
      for (size_t batch_i = 0; batch_i < Data.numBatches(); batch_i++) {
        for (size_t vec_i = 0; vec_i < Data[batch_i]._batch_size; vec_i++) {
          std::getline(file, line_from_file);
          ASSERT_EQ(format_vector_as_svm_line(Data[batch_i], vec_i),
                    line_from_file);
        }
      }
      while (Data.numBatches() == bn) {
        Data.loadNextBatchSet();
        for (size_t batch_i = 0; batch_i < Data.numBatches(); batch_i++) {
          for (size_t vec_i = 0; vec_i < Data[batch_i]._batch_size; vec_i++) {
            std::getline(file, line_from_file);
            ASSERT_EQ(format_vector_as_svm_line(Data[batch_i], vec_i),
                      line_from_file);
          }
        }
      }
      file.seekg(0);
    }
  }
  file.close();
}