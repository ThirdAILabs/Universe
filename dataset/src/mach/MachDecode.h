#pragma once

#include "MachIndex.h"
#include <bolt_vector/src/BoltVector.h>
#include <vector>

namespace thirdai::dataset::mach {

// TODO(david): implement and test other decoding methods

/**
 * Given the output activations to a mach model, decode using the mach index
 * back to the original classes. We take the top K values from the output and
 * select a candidate list of documents based on the inverted index. For each
 * one of those candidates we compute a score by summing the activations of its
 * hashed indicies. TopKUnlimited means we sum from ALL hashed indices instead
 * of those just in the top K activations.
 */
std::vector<std::pair<std::string, double>> topKUnlimitedDecode(
    const BoltVector& output, const MachIndexPtr& index,
    uint32_t min_num_eval_results, uint32_t top_k_per_eval_aggregation);

}  // namespace thirdai::dataset::mach