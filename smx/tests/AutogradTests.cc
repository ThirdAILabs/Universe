#include <gtest/gtest.h>
#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/DenseTensor.h>

namespace thirdai::smx::tests {

VariablePtr func(const std::vector<VariablePtr>& inputs,
                 const std::string& name, std::vector<std::string>& call_log) {
  auto out = DenseTensor::make(Shape(1UL, 2UL), Dtype::f32);

  GradFunc grad_func = [name, &call_log, inputs](
                           const TensorPtr& grad,
                           const std::vector<VariablePtr>& grad_inputs) {
    (void)grad;

    EXPECT_EQ(grad_inputs, inputs);
    call_log.push_back(name);
  };

  return Variable::make(out, grad_func, inputs);
}

VariablePtr input() {
  return Variable::make(DenseTensor::make(Shape(1UL, 2UL), Dtype::f32), true);
}

TEST(AutogradTests, SimpleGraph) {
  std::vector<std::string> call_log;

  auto in = input();

  auto out_1 = func({in}, "func_1", call_log);
  auto out_2 = func({in, out_1}, "func_2", call_log);
  auto out_3 = func({in, out_1, out_2}, "func_3", call_log);
  auto out_4 = func({in, out_1, out_2, out_3}, "func_4", call_log);
  auto out_5 = func({in, out_1, out_2, out_3, out_4}, "func_5", call_log);

  out_5->backpropagate(DenseTensor::make(Shape(1UL, 2UL), Dtype::f32));

  std::vector<std::string> expected_backprop_order = {
      "func_5", "func_4", "func_3", "func_2", "func_1",
  };

  ASSERT_EQ(call_log, expected_backprop_order);
}

TEST(AutogradTests, ComplexGraph) {
  std::vector<std::string> call_log;

  auto in_1 = input();
  auto in_2 = input();
  auto in_3 = input();

  auto out_1 = func({in_1, in_2}, "func_1", call_log);
  auto out_2 = func({in_2, out_1}, "func_2", call_log);
  auto out_3 = func({in_3, out_1, out_2}, "func_3", call_log);
  auto out_4 = func({in_3, out_3}, "func_4", call_log);
  auto out_5 = func({out_1, out_3, out_4}, "func_5", call_log);
  auto out_6 = func({out_3, out_5}, "func_6", call_log);
  auto out_7 = func({out_3, out_5, out_6}, "func_7", call_log);
  auto out_8 = func({out_4, out_7}, "func_8", call_log);

  out_8->backpropagate(DenseTensor::make(Shape(1UL, 2UL), Dtype::f32));

  std::vector<std::string> expected_backprop_order = {
      "func_8", "func_7", "func_6", "func_5",
      "func_4", "func_3", "func_2", "func_1",
  };

  ASSERT_EQ(call_log, expected_backprop_order);
}

}  // namespace thirdai::smx::tests