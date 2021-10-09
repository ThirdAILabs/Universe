#include "../utils/dataset/svm/SVMDataset.h"
#include "../utils/hashing/DensifiedMinHash.h"
#include "src/Flash.h"
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 5) {
    std::clog
        << ""
           "Test Flash in a limited way from C++. For now, this is chiefly \n"
           "a way to test Flash directly through C++ for performance \n"
           "optimization, and as such it only works with SVM files and \n"
           "DensifiedMinHash (for now)."
        << std::endl;
    std::clog << "Usage: ./run_flash <data_file> <query_file> <num_tables> "
                 "<hashes_per_table>"
              << std::endl;
    return -1;
  }

  uint32_t batch_size = 1000;
  uint32_t num_tables = atoi(argv[3]);
  uint32_t hashes_per_table = atoi(argv[4]);
  uint32_t top_k = 100;
  uint32_t range = 1000000;
  auto dataset = thirdai::utils::SVMDataset(argv[1], batch_size, 1);
  auto hash_func =
      thirdai::utils::DensifiedMinHash(hashes_per_table, num_tables, range);
  auto flash = thirdai::search::Flash<uint64_t>(hash_func);
  flash.addDataset(dataset);

  auto queries = thirdai::utils::SVMDataset(argv[2], UINT32_MAX, 1);
  queries.loadNextBatchSet();
  auto all_results = flash.queryBatch(queries[0], top_k, false);

  for (auto& result : all_results) {
    for (auto i : result) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }
}