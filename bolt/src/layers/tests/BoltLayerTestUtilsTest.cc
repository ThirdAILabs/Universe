#include "BoltLayerTestUtils.h"
#include <gtest/gtest.h>

namespace thirdai::bolt::tests {

TEST(BoltLayerTestUtilsTests, MatrixConstructor) {
  Matrix m({{1, 2, 3}, {4, 5, 6}});

  ASSERT_EQ(m.nRows(), 2);
  ASSERT_EQ(m.nCols(), 3);
  for (uint32_t i = 0; i < 2; i++) {
    for (uint32_t j = 0; j < 3; j++) {
      ASSERT_EQ(m(i, j), (j + 1) + i * 3);
    }
  }
}

TEST(BoltLayerTestUtilsTests, MatrixTranspose) {
  Matrix m({{1, 2, 3}, {4, 5, 6}});

  Matrix mt = m.transpose();
  ASSERT_EQ(mt.nRows(), 3);
  ASSERT_EQ(mt.nCols(), 2);
  for (uint32_t i = 0; i < 3; i++) {
    for (uint32_t j = 0; j < 2; j++) {
      ASSERT_EQ(mt(i, j), (i + 1) + j * 3);
      ASSERT_EQ(mt(i, j), m(j, i));
    }
  }

  Matrix mtt = mt.transpose();

  ASSERT_EQ(mtt.nRows(), m.nRows());
  ASSERT_EQ(mtt.nCols(), m.nCols());
  for (uint32_t i = 0; i < m.nRows(); i++) {
    for (uint32_t j = 0; j < m.nCols(); j++) {
      ASSERT_EQ(m(i, j), mtt(i, j));
    }
  }
}

TEST(BoltLayerTestUtilsTests, MatrixMultiplication1) {
  Matrix a({{1, 2, 3}, {4, 5, 6}});
  Matrix b({{3, 4}, {2, 1}, {7, 4}});

  Matrix ab = a.multiply(b);

  Matrix correct({{28, 18}, {64, 45}});

  ASSERT_EQ(ab.nRows(), correct.nRows());
  ASSERT_EQ(ab.nCols(), correct.nCols());
  for (uint32_t i = 0; i < correct.nRows(); i++) {
    for (uint32_t j = 0; j < correct.nCols(); j++) {
      ASSERT_EQ(ab(i, j), correct(i, j));
    }
  }

  Matrix at({{1, 4}, {2, 5}, {3, 6}});

  Matrix atb = at.transpose().multiply(b);

  ASSERT_EQ(atb.nRows(), correct.nRows());
  ASSERT_EQ(atb.nCols(), correct.nCols());
  for (uint32_t i = 0; i < correct.nRows(); i++) {
    for (uint32_t j = 0; j < correct.nCols(); j++) {
      ASSERT_EQ(atb(i, j), correct(i, j));
    }
  }
}

TEST(BoltLayerTestUtilsTests, MatrixMultiplication2) {
  Matrix a({{3, 5}, {8, 2}, {1, 9}});
  Matrix b({{3, 4, 2, 1}, {7, 4, 6, 3}});

  Matrix ab = a.multiply(b);

  Matrix correct({{44, 32, 36, 18}, {38, 40, 28, 14}, {66, 40, 56, 28}});

  ASSERT_EQ(ab.nRows(), correct.nRows());
  ASSERT_EQ(ab.nCols(), correct.nCols());
  for (uint32_t i = 0; i < correct.nRows(); i++) {
    for (uint32_t j = 0; j < correct.nCols(); j++) {
      ASSERT_EQ(ab(i, j), correct(i, j));
    }
  }

  Matrix at({{3, 8, 1}, {5, 2, 9}});

  Matrix atb = at.transpose().multiply(b);

  ASSERT_EQ(atb.nRows(), correct.nRows());
  ASSERT_EQ(atb.nCols(), correct.nCols());
  for (uint32_t i = 0; i < correct.nRows(); i++) {
    for (uint32_t j = 0; j < correct.nCols(); j++) {
      ASSERT_EQ(atb(i, j), correct(i, j));
    }
  }
}

TEST(BoltLayerTestUtilsTests, MatrixAdd) {
  Matrix a({{3, 5, 1}, {8, 2, 14}, {1, 9, 7}});
  Matrix b({{3, 7, -2}});

  a.add(b);

  Matrix correct({{6, 12, -1}, {11, 9, 12}, {4, 16, 5}});

  for (uint32_t i = 0; i < 3; i++) {
    for (uint32_t j = 0; j < 2; j++) {
      ASSERT_EQ(a(i, j), correct(i, j));
    }
  }
}

TEST(BoltLayerTestUtilsTests, SparseMatrix) {
  Matrix sparse(
      {{0, 1, 3}, {1, 3, 4}, {0, 4}, {2}, {0, 2, 4}},
      {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0}, {9.0}, {10.0, 11.0, 12.0}},
      5);

  Matrix dense({{1.0, 2.0, 0.0, 3.0, 0.0},
                {0.0, 4.0, 0.0, 5.0, 6.0},
                {7.0, 0.0, 0.0, 0.0, 8.0},
                {0.0, 0.0, 9.0, 0.0, 0.0},
                {10.0, 0.0, 11.0, 0.0, 12.0}});

  ASSERT_EQ(sparse.nRows(), dense.nRows());
  ASSERT_EQ(sparse.nCols(), dense.nCols());
  for (uint32_t i = 0; i < 5; i++) {
    for (uint32_t j = 0; j < 5; j++) {
      ASSERT_EQ(sparse(i, j), dense(i, j));
    }
  }
}

}  // namespace thirdai::bolt::tests