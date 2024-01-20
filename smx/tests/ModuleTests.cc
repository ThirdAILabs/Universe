#include <gtest/gtest.h>
#include <archive/tests/Utils.h>
#include <smx/src/modules/Linear.h>
#include <smx/src/modules/Module.h>
#include <memory>
#include <stdexcept>

namespace thirdai::smx::tests {

bool containsParam(const std::vector<VariablePtr>& vars, const VariablePtr& x) {
  return std::find(vars.begin(), vars.end(), x) != vars.end();
}

TEST(ModuleTests, ParameterDiscover) {
  auto l1 = std::make_shared<Linear>(2, 3);
  auto l2 = std::make_shared<Linear>(4, 5);
  auto l3 = std::make_shared<Linear>(6, 7);

  auto seq1 = std::make_shared<Sequential>();
  seq1->append(l1);
  seq1->append(l2);

  auto seq2 = std::make_shared<Sequential>();
  seq2->append(l1);
  seq2->append(seq1);
  seq2->append(l3);

  auto params1 = seq1->parameters();
  ASSERT_EQ(params1.size(), 4);
  ASSERT_TRUE(containsParam(params1, l1->weight()));
  ASSERT_TRUE(containsParam(params1, l1->bias()));
  ASSERT_TRUE(containsParam(params1, l2->weight()));
  ASSERT_TRUE(containsParam(params1, l2->bias()));

  auto params2 = seq2->parameters();
  ASSERT_EQ(params2.size(), 6);
  ASSERT_TRUE(containsParam(params2, l1->weight()));
  ASSERT_TRUE(containsParam(params2, l1->bias()));
  ASSERT_TRUE(containsParam(params2, l2->weight()));
  ASSERT_TRUE(containsParam(params2, l2->bias()));
  ASSERT_TRUE(containsParam(params2, l3->weight()));
  ASSERT_TRUE(containsParam(params2, l3->bias()));
}

TEST(ModuleTests, RegisteringModuleWithItself) {
  auto l = std::make_shared<Linear>(2, 3);

  auto seq = std::make_shared<Sequential>();
  seq->append(l);

  CHECK_EXCEPTION(seq->append(seq), "Cannot register a module with itself.",
                  std::runtime_error)
}

TEST(ModuleTests, RegisteringParentModuleWithChild) {
  auto l = std::make_shared<Linear>(2, 3);

  auto seq1 = std::make_shared<Sequential>();
  seq1->append(l);

  auto seq2 = std::make_shared<Sequential>();
  seq2->append(l);
  seq2->append(seq1);

  auto seq3 = std::make_shared<Sequential>();
  seq3->append(l);
  seq3->append(seq2);

  CHECK_EXCEPTION(
      seq1->append(seq3),
      "Cannot register module as it contains the module it is being "
      "registered with as a submodule.",
      std::runtime_error)
}

}  // namespace thirdai::smx::tests