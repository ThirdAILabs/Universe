#include "TensorManipulation.h"
#include <smx/src/tensor/Functions.h>

namespace thirdai::smx {

VariablePtr transpose(const VariablePtr& input,
                      const std::vector<size_t>& perm) {
  std::vector<size_t> reverse_perm(perm.size());

  for (size_t i = 0; i < perm.size(); i++) {
    /**
     * If perm[i] = j then that means that the i-th dimension of the output is
     * the j-th dimension of the input. Thus to undo the transpose we need
     * reverse_perm[j] = i in order to map the i-th dimension of the output back
     * to the j-th dimension of the input.
     *
     * We can think about perm[i] = j as meaning j -> i. Thus defining the
     * reverse permutation as reverse_perm[j] = i means that i -> j, effectively
     * undoing the original transpose.
     */
    reverse_perm[perm[i]] = i;
  }

  GradFunc grad_func = [reverse_perm](const TensorPtr& grad,
                                      const std::vector<VariablePtr>& inputs) {
    if (inputs[0]->requiresGrad()) {
      inputs[0]->addGradient(transpose(grad, reverse_perm));
    }
  };

  return Variable::make(transpose(input->tensor(), perm), grad_func, {input});
}

VariablePtr reshape(const VariablePtr& input, const Shape& new_shape) {
  Shape original_shape = input->tensor()->shape();

  GradFunc grad_func = [original_shape](
                           const TensorPtr& grad,
                           const std::vector<VariablePtr>& inputs) {
    if (inputs[0]->requiresGrad()) {
      inputs[0]->addGradient(reshape(grad, original_shape));
    }
  };

  return Variable::make(reshape(input->tensor(), new_shape), grad_func,
                        {input});
}

}  // namespace thirdai::smx