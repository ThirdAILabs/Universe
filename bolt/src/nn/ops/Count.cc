#include "Count.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>

namespace thirdai::bolt::nn::ops {

std::string nextCountOpName() {
  static uint32_t constructed = 0;
  return "count_" + std::to_string(++constructed);
}

std::shared_ptr<Count> Count::make() {
  return std::shared_ptr<Count>(new Count());
}

Count::Count() : Op(nextCountOpName()) {}

void Count::forward(const autograd::ComputationList& inputs,
                    tensor::TensorPtr& output, uint32_t index_in_batch,
                    bool training) {
  (void)training;
  assert(inputs.size() == 1);
  uint32_t count = inputs[0]->tensor()->getVector(index_in_batch).len;
  output->getVector(index_in_batch).active_neurons[0] = count;
  output->getVector(index_in_batch).activations[0] = 1.0;
}

void Count::summary(std::ostream& summary,
                    const autograd::ComputationList& inputs,
                    const autograd::Computation* output) const {
  summary << "Count(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name();
}

autograd::ComputationPtr Count::apply(autograd::ComputationPtr input) {
  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void Count::save(cereal::BinaryOutputArchive&) const;

template <class Archive>
void Count::save(Archive& archive) const {
  archive(cereal::base_class<Op>(this));
}

template void Count::load(cereal::BinaryInputArchive&);

template <class Archive>
void Count::load(Archive& archive) {
  archive(cereal::base_class<Op>(this));
}

}  // namespace thirdai::bolt::nn::ops

namespace cereal {

/**
 * This is because the Op base class only uses a serialize function, whereas
 * this Op uses a load/save pair. This tells cereal to use the load save pair
 * instead of the serialize method of the parent class. See docs here:
 * https://uscilab.github.io/cereal/serialization_functions.html#inheritance
 */
template <class Archive>
struct specialize<Archive, thirdai::bolt::nn::ops::Count,
                  cereal::specialization::member_load_save> {};

}  // namespace cereal

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::Count)