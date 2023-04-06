#pragma once

#include "MachIndex.h"
#include <bolt_vector/src/BoltVector.h>
#include <vector>

namespace thirdai::dataset::mach {

// TODO(david): implement and test other decoding methods

/**
 * Given the output activations to a mach model, decode using the mach index
 * back to the original classes.
 * TODO(david) implement the more efficient version.
 */
std::vector<std::pair<std::string, double>> topKUnlimitedDecode(
    const BoltVector& output, const MachIndexPtr& index,
    uint32_t min_num_eval_results, uint32_t top_k_per_eval_aggregation);

}  // namespace thirdai::dataset::mach