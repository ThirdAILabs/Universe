#pragma once

#include <bolt/src/nn/model/Model.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::train {
class DistributedComm {

 public:
  virtual void communicate(const bolt::nn::model::ModelPtr& model) = 0;

  virtual uint64_t minNumBatches(uint64_t num_batches) = 0;

  virtual ~DistributedComm() = default;
};

using DistributedCommPtr = std::shared_ptr<DistributedComm>;

}  // namespace thirdai::bolt::train