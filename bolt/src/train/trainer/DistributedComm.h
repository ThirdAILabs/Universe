#pragma once

#include <bolt/src/nn/model/Model.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

class DistributedComm {
 public:
  virtual void communicate(const ModelPtr& model) = 0;

  virtual uint64_t minNumBatches(uint64_t num_batches) = 0;

  virtual std::vector<float> broadcastMetrics(std::vector<float>) = 0;

  virtual ~DistributedComm() = default;
};

using DistributedCommPtr = std::shared_ptr<DistributedComm>;

}  // namespace thirdai::bolt