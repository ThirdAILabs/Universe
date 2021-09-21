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

// Downloaded by bin/get_datsets.sh from
// https://www.csie.ntu.edu.tw/~cjlin/lisbsvmtools/datasets/multilabel/bibtex.bz2
// Downloaded file is located in build/utils/tests/dataset/svm/, the same
// directory as the exedcutable for this file.
static std::string filename = "bibtex";

/**
 * Formats a vector from a batch as a line in SVM format.
 */
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
  for (size_t nonzero_i = 1; nonzero_i < batch._lens[vec_i]; nonzero_i++) {
    line << " " << batch._indices[vec_i][nonzero_i] << ":"
         << batch._values[vec_i][nonzero_i];
  }
  return line.str();
}

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

/**
 * Helper function called in TEST(SVMDatasetTest, BatchSizeAndNumber)
 * Evaluates the correctness of a batch set after a successful call to
 * SVMDataset::loadNextBatchSet() Particularly, check:
 *  - The number of times loadNextBatchSet() is called.
 *  - The number of batches loaded each time loadNextBatchSet() is called.
 *  - The size of each batch.
 */
static void evaluate_load(SVMDataset& Data, uint64_t target_batch_size,
                          uint64_t target_batch_number,
                          uint64_t number_of_times_loaded, uint64_t vec_num) {
  if (target_batch_number > 0) {
    if (Data.numBatches() > target_batch_number) {
      std::cout << "Num batches is greater than target batch number. Something "
                   "is terribly wrong."
                << std::endl;
    }
    ASSERT_LE(Data.numBatches(), target_batch_number);
  }
  uint64_t expected_num_batches = get_expected_num_batches(
      target_batch_size, target_batch_number, number_of_times_loaded, vec_num);
  if (Data.numBatches() != expected_num_batches) {
    std::cout << "Num batches expected: " << expected_num_batches
              << " got: " << Data.numBatches() << std::endl
              << " Config: bn = " << target_batch_number
              << " bs = " << target_batch_size
              << " successful loads = " << number_of_times_loaded << std::endl;
  }
  ASSERT_EQ(Data.numBatches(), expected_num_batches);
  for (size_t batch_i = 0; batch_i < Data.numBatches(); batch_i++) {
    ASSERT_LE(Data[batch_i]._batch_size, target_batch_size);
    uint64_t expected_batch_size =
        get_expected_batch_size(target_batch_size, target_batch_number,
                                number_of_times_loaded, vec_num, batch_i);
    if (Data[batch_i]._batch_size != expected_batch_size) {
      std::cout << "Batch size expected: " << expected_batch_size
                << " got: " << Data[batch_i]._batch_size << std::endl
                << " Config: bn = " << target_batch_number
                << " bs = " << target_batch_size
                << " successful loads = " << number_of_times_loaded
                << " batch_i = " << batch_i << std::endl;
    }
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
  // count number of vectors (non-whitespace lines) in the file.
  std::ifstream _file(filename);
  std::string line;
  uint64_t expected_vec_num = 0;
  while (std::getline(_file, line)) {
    if (line.find_first_not_of(" \t\n\v\f\r") != std::string::npos) {
      expected_vec_num++;
    }
  }
  _file.close();
  std::cout << "Expect " << expected_vec_num << " vectors in this dataset."
            << std::endl;

  uint64_t batch_sizes[] = {150, 100, 50, 30};
  uint64_t batch_nums[] = {150, 100, 50, 0};

  for (auto bs : batch_sizes) {
    for (auto bn : batch_nums) {
      SVMDataset Data(filename, bs, bn);
      size_t successful_loads = 0;

      Data.loadNextBatchSet();
      successful_loads++;
      evaluate_load(Data, bs, bn, successful_loads, expected_vec_num);

      while (Data.numBatches() > 0) {
        Data.loadNextBatchSet();
        if (Data.numBatches() > 0) {
          successful_loads++;
          evaluate_load(Data, bs, bn, successful_loads, expected_vec_num);
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

/**
 * Loads SVM dataset with different target batch size and batch number
 * combinations, converts each configuration back to an SVM file, and checks
 * that the rewritten file is the same as the initial file.
 */
TEST(SVMDatasetTest, CompareRewrittenFile) {
  std::ifstream file(filename);
  std::string line_from_file;
  uint64_t line_num = 0;

  uint64_t batch_sizes[] = {50, 100, 150};
  uint64_t batch_nums[] = {0, 50, 100, 150};
  for (auto bs : batch_sizes) {
    for (auto bn : batch_nums) {
      SVMDataset Data(filename, bs, bn);

      Data.loadNextBatchSet();
      for (size_t batch_i = 0; batch_i < Data.numBatches(); batch_i++) {
        for (size_t vec_i = 0; vec_i < Data[batch_i]._batch_size; vec_i++) {
          std::getline(file, line_from_file);
          std::string vector_svm_line =
              format_vector_as_svm_line(Data[batch_i], vec_i);
          if (vector_svm_line != line_from_file) {
            std::cout << "Line " << line_num
                      << " is different from original file:" << std::endl;
            std::cout << "Original line: " << line_from_file << std::endl;
            std::cout << "Processed line: " << vector_svm_line << std::endl;
          }
          ASSERT_EQ(vector_svm_line, line_from_file);
          line_num++;
        }
      }
      while (bn != 0 && Data.numBatches() == bn) {
        Data.loadNextBatchSet();
        for (size_t batch_i = 0; batch_i < Data.numBatches(); batch_i++) {
          for (size_t vec_i = 0; vec_i < Data[batch_i]._batch_size; vec_i++) {
            std::getline(file, line_from_file);
            std::string vector_svm_line =
                format_vector_as_svm_line(Data[batch_i], vec_i);
            if (vector_svm_line != line_from_file) {
              std::cout << "Line " << line_num << std::endl;
            }
            ASSERT_EQ(vector_svm_line, line_from_file);
            line_num++;
          }
        }
      }
      file.seekg(0);
    }
  }
  file.close();
}