#pragma once

#include <bolt/src/nn/model/Model.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::train {
class DistributedCommInterface {
 public:
  virtual void communicate(const bolt::nn::model::ModelPtr& model) {
    (void)model;
  }

  virtual uint64_t min_num_batches(uint64_t num_batches) {
    (void)num_batches;
    throw std::runtime_error("Method min_num_batches not implemented.");
  }
  virtual ~DistributedCommInterface() = default;
};

using DistributedCommInterfacePtr = std::shared_ptr<DistributedCommInterface>;

}  // namespace thirdai::bolt::train