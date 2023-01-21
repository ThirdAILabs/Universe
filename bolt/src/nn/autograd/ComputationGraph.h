#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::autograd {

/**
 * Returns all non input computations in the order that they should be executed
 * in the forward pass such that all inputs to a computation are executed before
 * the given computation is executed. Addidtionally the ordering is such that if
 * the sequence is traversed in reverse during backpropagation all of the
 * computations which use the output of a given computation will have
 * backpropagate called before the given computation is reached. Checks that all
 * inputs are used by a computation and that no computations depend on inputs
 * that are not provided in the list of inputs.
 */
ComputationList getComputationOrder(const ComputationList& inputs,
                                    const ComputationList& outputs);

/**
 * Returns a map of computations to how many computations they are used in. This
 * is a helper function for getComputationOrder used to ensure that computations
 * are scheduled before all of their dependents. It is also used in the model to
 * check various properties of the graph.
 */
std::unordered_map<ComputationPtr, uint32_t> countDependentComputations(
    const ComputationList& outputs);

}  // namespace thirdai::bolt::nn::autograd