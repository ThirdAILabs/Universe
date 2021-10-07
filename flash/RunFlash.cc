#include "../utils/dataset/svm/SVMDataset.h"
#include "../utils/hashing/DensifiedMinHash.h"
#include "src/Flash.h"

int main() {
  auto dataset = thirdai::utils::SVMDataset("/Users/josh/webspam_data.svm", 1000, 1);
  auto hash_func = thirdai::utils::DensifiedMinHash(2, 100);
  auto flash = thirdai::search::Flash<uint64_t>(hash_func);
  flash.addDataset(dataset);

  auto queries = thirdai::utils::SVMDataset("/Users/josh/webspam_queries.svm", UINT32_MAX, 1);
  queries.loadNextBatchSet();
  auto all_results = flash.queryBatch(queries[0], 100, true);

  for (auto &result : all_results) {
    for (auto i : result) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }
}