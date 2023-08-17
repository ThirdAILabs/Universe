
#include <bolt/src/layers/GradientClipper.h>
#include <gtest/gtest.h>
#include <_types/_uint32_t.h>
namespace thirdai::bolt::tests {

std::vector<float> gradients = {-5.0, -4.0, -3.0, -2.0, -1.0, 0,
                                1.0,  2.0,  3.0,  4.0,  5.0};

float compute_norm(const std::vector<float>& gradients) {
  return std::sqrt(
      std::accumulate(gradients.begin(), gradients.end(), 0.0,
                      [](float sum, float val) { return sum + val * val; }));
}

TEST(GradientClippingTests, ValueTest) {
  auto gradient_clipper = std::make_shared<GradientClipperByValue>(4.0);

  auto final_gradients = gradient_clipper->clipVector(gradients);

  for (float final_gradient : final_gradients) {
    ASSERT_LE(final_gradient, 4);
  }
}

TEST(GradientClippingTests, NormTest) {
  auto gradient_clipper = std::make_shared<GradientClipperByNorm>(4.0);

  auto final_gradients = gradient_clipper->clipVector(gradients);

  float norm = compute_norm(gradients);
  ASSERT_GE(norm, 4.0);

  float factor = 4.0 / norm;

  for (uint32_t i = 0; i < final_gradients.size(); i++) {
    ASSERT_EQ(final_gradients[i], gradients[i] * factor);
  }
}

TEST(GradientClippingTests, FractionTest) {
  auto gradient_clipper = std::make_shared<GradientClipperByFraction>(0.2);

  auto final_gradients = gradient_clipper->clipVector(gradients);

  for (float final_gradient : final_gradients) {
    ASSERT_LE(final_gradient, 4);
  }
}

}  // namespace thirdai::bolt::tests