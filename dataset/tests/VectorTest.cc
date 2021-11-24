#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/Vectors.h>

namespace thirdai::dataset {

TEST(VectorTest, DenseVectorMemoryStartingOwningMemory) {
  const uint32_t vec_dim = 100;
  thirdai::dataset::DenseVector a(vec_dim);
  for (uint32_t i = 0; i < vec_dim; i++) {
    a._values[i] = i;
  }
  ASSERT_TRUE(a.owns_data());

  thirdai::dataset::DenseVector b = std::move(a);  // Move
  ASSERT_EQ(vec_dim, b.dim());
  ASSERT_TRUE(b.owns_data());
  for (uint32_t i = 0; i < vec_dim; i++) {
    ASSERT_EQ(b._values[i], i);
  }

  thirdai::dataset::DenseVector c(b);  // Copy
  ASSERT_TRUE(b.owns_data());
  ASSERT_TRUE(c.owns_data());
  ASSERT_NE(b._values, c._values);
  ASSERT_EQ(b.dim(), vec_dim);
  ASSERT_EQ(c.dim(), vec_dim);
  for (uint32_t i = 0; i < vec_dim; i++) {
    ASSERT_EQ(b._values[i], i);
    ASSERT_EQ(c._values[i], i);
  }
}

TEST(VectorTest, DenseVectorMemoryStartingNotOwningMemory) {
  const uint32_t vec_dim = 5;
  float data[vec_dim];
  thirdai::dataset::DenseVector a(vec_dim, data, false);
  for (uint32_t i = 0; i < vec_dim; i++) {
    a._values[i] = i;
  }
  ASSERT_FALSE(a.owns_data());

  thirdai::dataset::DenseVector b = std::move(a);  // Move
  ASSERT_EQ(vec_dim, b.dim());
  ASSERT_FALSE(b.owns_data());
  for (uint32_t i = 0; i < vec_dim; i++) {
    ASSERT_EQ(b._values[i], i);
  }
  ASSERT_EQ(b._values, data);

  thirdai::dataset::DenseVector c(b);  // Copy
  ASSERT_FALSE(b.owns_data());
  ASSERT_TRUE(c.owns_data());
  ASSERT_NE(b._values, c._values);
  ASSERT_EQ(b.dim(), vec_dim);
  ASSERT_EQ(c.dim(), vec_dim);
  for (uint32_t i = 0; i < vec_dim; i++) {
    ASSERT_EQ(b._values[i], i);
    ASSERT_EQ(c._values[i], i);
  }
}

TEST(VectorTest, SparseVectorMemoryStartingOwningMemory) {
  const uint32_t vec_length = 100;
  thirdai::dataset::SparseVector a(vec_length);
  for (uint32_t i = 0; i < vec_length; i++) {
    a._values[i] = i;
    a._indices[i] = i;
  }
  ASSERT_TRUE(a.owns_data());

  thirdai::dataset::SparseVector b = std::move(a);  // Move
  ASSERT_EQ(vec_length, b.length());
  ASSERT_TRUE(b.owns_data());
  for (uint32_t i = 0; i < vec_length; i++) {
    ASSERT_EQ(b._values[i], i);
    ASSERT_EQ(b._indices[i], i);
  }

  thirdai::dataset::SparseVector c(b);  // Copy
  ASSERT_TRUE(b.owns_data());
  ASSERT_TRUE(c.owns_data());
  ASSERT_NE(b._values, c._values);
  ASSERT_NE(b._indices, c._indices);
  ASSERT_EQ(b.length(), vec_length);
  ASSERT_EQ(c.length(), vec_length);
  for (uint32_t i = 0; i < vec_length; i++) {
    ASSERT_EQ(b._values[i], i);
    ASSERT_EQ(b._indices[i], i);
    ASSERT_EQ(c._values[i], i);
    ASSERT_EQ(c._indices[i], i);
  }
}

TEST(VectorTest, SparseVectorMemoryStartingNotOwningMemory) {
  const uint32_t vec_length = 100;
  uint32_t indices[vec_length];
  float values[vec_length];
  thirdai::dataset::SparseVector a(indices, values, vec_length, false);
  for (uint32_t i = 0; i < vec_length; i++) {
    a._values[i] = i;
    a._indices[i] = i;
  }
  ASSERT_FALSE(a.owns_data());

  thirdai::dataset::SparseVector b = std::move(a);  // Move
  ASSERT_EQ(vec_length, b.length());
  ASSERT_FALSE(b.owns_data());
  for (uint32_t i = 0; i < vec_length; i++) {
    ASSERT_EQ(b._values[i], i);
    ASSERT_EQ(b._indices[i], i);
  }
  ASSERT_EQ(b._values, values);
  ASSERT_EQ(b._indices, indices);

  thirdai::dataset::SparseVector c(b);  // Copy
  ASSERT_FALSE(b.owns_data());
  ASSERT_TRUE(c.owns_data());
  ASSERT_NE(b._values, c._values);
  ASSERT_NE(b._indices, c._indices);
  ASSERT_EQ(b.length(), vec_length);
  ASSERT_EQ(c.length(), vec_length);
  for (uint32_t i = 0; i < vec_length; i++) {
    ASSERT_EQ(b._values[i], i);
    ASSERT_EQ(b._indices[i], i);
    ASSERT_EQ(c._values[i], i);
    ASSERT_EQ(c._indices[i], i);
  }
}

}  // namespace thirdai::dataset