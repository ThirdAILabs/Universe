#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt {

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
                                    const ComputationList& outputs,
                                    const std::vector<LossPtr>& losses);

/**
 * Returns a map of computations to how many computations they are used in. This
 * is a helper function for getComputationOrder used to ensure that computations
 * are scheduled before all of their dependents. It is also used in the model to
 * check various properties of the graph. It starts at the loss functions and
 * the computations they are applied to and works backward through the graph.
 */
std::unordered_map<ComputationPtr, uint32_t> countDependentComputations(
    const std::vector<LossPtr>& losses);

/**
 * Returns all the computations that are used in loss functions. Throws if a
 * computation is used by multiple loss functions.
 */
ComputationList computationsUsedInLossFunctions(
    const std::vector<LossPtr>& losses);

/**
 * Checks that loss functions only apply to computations that have no dependent
 * computations. Called in getComputationOrder.
 */
void checkLossesOnlyApplyToTerminalOutputs(const std::vector<LossPtr>& losses);

/**
 * Check that the user specifed outputs of hte model are all present in the
 * final computation order. Called in getComputationOrder.
 */
void checkAllOutputsInComputationOrder(const ComputationList& computation_order,
                                       const ComputationList& outputs);

}  // namespace thirdai::bolt